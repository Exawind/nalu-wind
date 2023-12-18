// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "AMSAlgDriver.h"
#include "master_element/MasterElementRepo.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_utils/NgpFieldManager.h"
#include "ngp_algorithms/MetricTensorElemAlg.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_io/IossBridge.hpp>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>

namespace sierra {
namespace nalu {

class Realm;

AMSAlgDriver::AMSAlgDriver(Realm& realm)
  : realm_(realm),
    avgVelocity_(NULL),
    avgVelocityRTM_(NULL),
    avgTkeResolved_(NULL),
    avgDudx_(NULL),
    metric_(NULL),
    beta_(NULL),
    resAdequacy_(NULL),
    avgResAdequacy_(NULL),
    avgProduction_(NULL),
    avgTime_(NULL),
    avgMdot_(NULL),
    forcingComp_(NULL),
    metricTensorAlgDriver_(realm_, "metric_tensor"),
    avgMdotAlg_(realm_),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    resetAMSAverages_(realm_.solutionOptions_->resetAMSAverages_)
{
  if (!realm.realmUsesEdges_)
    throw std::runtime_error("AMS not supported on element runs.");
  if (turbulenceModel_ != TurbulenceModel::SST_AMS) {
    throw std::runtime_error(
      "User has requested AMS, however, turbulence model has not been set "
      "to sst_ams, the only one supported by this driver currently.");
  }
}

void
AMSAlgDriver::register_nodal_fields(const stk::mesh::PartVector& part_vec)
{
  stk::mesh::MetaData& meta = realm_.meta_data();
  const int nDim = meta.spatial_dimension();
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);

  // Set numStates as 2, so avg quantities can be updated through Picard iters
  const int numStates = 2;

  // Nodal fields
  beta_ = &(meta.declare_field<double>(stk::topology::NODE_RANK, "k_ratio"));
  stk::mesh::put_field_on_mesh(*beta_, selector, nullptr);

  avgVelocity_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "average_velocity", numStates));
  stk::mesh::put_field_on_mesh(*avgVelocity_, selector, nDim, nullptr);
  stk::io::set_field_output_type(
    *avgVelocity_, stk::io::FieldOutputType::VECTOR_3D);
  realm_.augment_restart_variable_list("average_velocity");

  if (
    realm_.solutionOptions_->meshMotion_ ||
    realm_.solutionOptions_->externalMeshDeformation_) {
    avgVelocityRTM_ = &(meta.declare_field<double>(
      stk::topology::NODE_RANK, "average_velocity_rtm"));
    stk::mesh::put_field_on_mesh(*avgVelocityRTM_, selector, nDim, nullptr);
    stk::io::set_field_output_type(
      *avgVelocityRTM_, stk::io::FieldOutputType::VECTOR_3D);
  }

  avgProduction_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "average_production", numStates));
  stk::mesh::put_field_on_mesh(*avgProduction_, selector, nullptr);
  realm_.augment_restart_variable_list("average_production");

  avgDudx_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "average_dudx", numStates));
  stk::mesh::put_field_on_mesh(*avgDudx_, selector, nDim * nDim, nullptr);
  realm_.augment_restart_variable_list("average_dudx");

  avgTkeResolved_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "average_tke_resolved", numStates));
  stk::mesh::put_field_on_mesh(*avgTkeResolved_, selector, nullptr);
  realm_.augment_restart_variable_list("average_tke_resolved");

  avgTime_ =
    &(meta.declare_field<double>(stk::topology::NODE_RANK, "rans_time_scale"));
  stk::mesh::put_field_on_mesh(*avgTime_, selector, nullptr);

  metric_ =
    &(meta.declare_field<double>(stk::topology::NODE_RANK, "metric_tensor"));
  stk::mesh::put_field_on_mesh(*metric_, selector, nDim * nDim, nullptr);

  resAdequacy_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*resAdequacy_, selector, nullptr);

  avgResAdequacy_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "avg_res_adequacy_parameter", numStates));
  stk::mesh::put_field_on_mesh(*avgResAdequacy_, selector, nullptr);
  realm_.augment_restart_variable_list("avg_res_adequacy_parameter");

  forcingComp_ = &(meta.declare_field<double>(
    stk::topology::NODE_RANK, "forcing_components", numStates));
  stk::mesh::put_field_on_mesh(*forcingComp_, selector, nDim, nullptr);
  stk::io::set_field_output_type(
    *forcingComp_, stk::io::FieldOutputType::VECTOR_3D);
}

void
AMSAlgDriver::register_edge_fields(const stk::mesh::PartVector& part_vec)
{
  stk::mesh::Selector selector = stk::mesh::selectUnion(part_vec);
  stk::mesh::MetaData& meta = realm_.meta_data();
  NaluEnv::self().naluOutputP0()
    << "Edge Mdot average added in AMS " << std::endl;
  avgMdot_ = &(meta.declare_field<double>(
    stk::topology::EDGE_RANK, "average_mass_flow_rate"));
  stk::mesh::put_field_on_mesh(*avgMdot_, selector, nullptr);
  realm_.augment_restart_variable_list("average_mass_flow_rate");
}

void
AMSAlgDriver::register_interior_algorithm(stk::mesh::Part* part)
{
  const AlgorithmType algType = INTERIOR;

  metricTensorAlgDriver_.register_elem_algorithm<MetricTensorElemAlg>(
    algType, part, "metric_tensor");

  if (!avgAlg_) {
    switch (realm_.solutionOptions_->turbulenceModel_) {
    case TurbulenceModel::SST_AMS:
      avgAlg_.reset(new SSTAMSAveragesAlg(realm_, part));
      break;
    default:
      throw std::runtime_error("AMSAlgDriver: non-supported turb model");
    }
  } else {
    avgAlg_->partVec_.push_back(part);
  }

  // avgMdot algorithm
  avgMdotAlg_.register_edge_algorithm<AMSAvgMdotEdgeAlg>(
    algType, part, "ams_avg_mdot_edge");
}

void
AMSAlgDriver::initial_work()
{
  compute_metric_tensor();

  // Initialize average_velocity, avg_dudx
  // We don't want to do this on restart where AMS fields are present
  if (resetAMSAverages_) {
    const auto& meta = realm_.meta_data();
    const auto& ngpMesh = realm_.ngp_mesh();
    const auto& fieldMgr = realm_.ngp_field_manager();
    const stk::mesh::Selector sel =
      (meta.locally_owned_part() | meta.globally_shared_part() |
       meta.aura_part()) &
      stk::mesh::selectField(*avgVelocity_);
    const int nDim = meta.spatial_dimension();

    // Copy velocity to average velocity
    auto& avgU = fieldMgr.get_field<double>(
      avgVelocity_->field_of_state(stk::mesh::StateNP1)
        .mesh_meta_data_ordinal());
    const unsigned velocityID = get_field_ordinal(meta, "velocity");
    const auto& U = fieldMgr.get_field<double>(velocityID);
    nalu_ngp::field_copy(ngpMesh, sel, avgU, U, nDim);

    // Copy dudx to average dudx
    auto avgDudx =
      fieldMgr.get_field<double>(avgDudx_->mesh_meta_data_ordinal());
    const auto& dudx =
      fieldMgr.get_field<double>(get_field_ordinal(meta, "dudx"));
    nalu_ngp::field_copy(ngpMesh, sel, avgDudx, dudx, nDim * nDim);
  }
}

void
AMSAlgDriver::initial_production()
{
  using Traits = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>;

  // Initialize average_production (after tvisc)
  // We don't want to do this on restart where AMS fields are present
  if (resetAMSAverages_) {

    const auto& meta = realm_.meta_data();
    const auto& ngpMesh = realm_.ngp_mesh();
    const auto& fieldMgr = realm_.ngp_field_manager();
    const stk::mesh::Selector sel =
      (meta.locally_owned_part() | meta.globally_shared_part() |
       meta.aura_part()) &
      stk::mesh::selectField(*avgVelocity_);
    const int nDim = meta.spatial_dimension();

    const auto tvisc = fieldMgr.get_field<double>(
      get_field_ordinal(meta, "turbulent_viscosity"));
    auto avgDudx =
      fieldMgr.get_field<double>(avgDudx_->mesh_meta_data_ordinal());

    // Compute average production
    auto avgProd =
      fieldMgr.get_field<double>(avgProduction_->mesh_meta_data_ordinal());
    nalu_ngp::run_entity_algorithm(
      "AMSAlgDriver_avgProd", ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
        NALU_ALIGNED DblType tij[nalu_ngp::NDimMax * nalu_ngp::NDimMax];
        for (int i = 0; i < nDim; ++i) {
          for (int j = 0; j < nDim; ++j) {
            const DblType avgSij = 0.5 * (avgDudx.get(mi, i * nDim + j) +
                                          avgDudx.get(mi, j * nDim + i));
            tij[i * nDim + j] = 2.0 * tvisc.get(mi, 0) * avgSij;
          }
        }

        NALU_ALIGNED DblType Pij[nalu_ngp::NDimMax * nalu_ngp::NDimMax];
        for (int i = 0; i < nDim; ++i) {
          for (int j = 0; j < nDim; ++j) {
            Pij[i * nDim + j] = 0.0;
            for (int m = 0; m < nDim; ++m) {
              Pij[i * nDim + j] +=
                avgDudx.get(mi, i * nDim + m) * tij[j * nDim + m] +
                avgDudx.get(mi, j * nDim + m) * tij[i * nDim + m];
            }
            Pij[i * nDim + j] *= 0.5;
          }
        }

        DblType instProd = 0.0;
        for (int i = 0; i < nDim; ++i)
          instProd += Pij[i * nDim + i];

        avgProd.get(mi, 0) = instProd;
      });

    avgProd.modify_on_device();
  }
}

void
AMSAlgDriver::initial_mdot()
{
  // Initialize mdot
  if (resetAMSAverages_) {

    const auto& meta = realm_.meta_data();
    const auto& ngpMesh = realm_.ngp_mesh();
    const auto& fieldMgr = realm_.ngp_field_manager();

    auto& avgMdot = fieldMgr.get_field<double>(get_field_ordinal(
      meta, "average_mass_flow_rate", stk::topology::EDGE_RANK));
    const auto& massFlowRate = fieldMgr.get_field<double>(
      get_field_ordinal(meta, "mass_flow_rate", stk::topology::EDGE_RANK));

    const stk::mesh::Selector sel =
      (meta.locally_owned_part() | meta.globally_shared_part()) &
      stk::mesh::selectField(
        *meta.get_field(stk::topology::EDGE_RANK, "average_mass_flow_rate"));

    nalu_ngp::field_copy(
      ngpMesh, sel, avgMdot, massFlowRate, 1, stk::topology::EDGE_RANK);
  }
}

void
AMSAlgDriver::execute()
{
  avgAlg_->execute();

  if (realm_.hasOverset_) {
    realm_.overset_field_update(avgVelocity_, 1, 3, false);
    realm_.overset_field_update(avgProduction_, 1, 1, false);
    realm_.overset_field_update(avgTkeResolved_, 1, 1, false);
    realm_.overset_field_update(avgResAdequacy_, 1, 1, false);
    realm_.overset_field_update(avgDudx_, 3, 3, false);
  }

  if (
    realm_.solutionOptions_->meshMotion_ ||
    realm_.solutionOptions_->externalMeshDeformation_) {
    realm_.compute_vrtm("average_velocity");
  }
  avgMdotAlg_.execute();
}

void
AMSAlgDriver::compute_metric_tensor()
{
  metricTensorAlgDriver_.execute();
}

void
AMSAlgDriver::post_iter_work()
{
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();

  auto ngpForcingComp =
    fieldMgr.get_field<double>(forcingComp_->mesh_meta_data_ordinal());

  ngpForcingComp.sync_to_host();

  VectorFieldType* forcingComp =
    meta.get_field<double>(stk::topology::NODE_RANK, "forcing_components");

  stk::mesh::copy_owned_to_shared(bulk, {forcingComp});
  if (realm_.hasPeriodic_) {
    realm_.periodic_delta_solution_update(forcingComp, 3);
  }
}

void
AMSAlgDriver::predict_state()
{
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();
  auto& avgVelN = fieldMgr.get_field<double>(
    avgVelocity_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& avgVelNp1 = fieldMgr.get_field<double>(
    avgVelocity_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());
  auto& avgDudxN = fieldMgr.get_field<double>(
    avgDudx_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& avgDudxNp1 = fieldMgr.get_field<double>(
    avgDudx_->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal());
  auto& avgProdN = fieldMgr.get_field<double>(
    avgProduction_->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal());
  auto& avgProdNp1 = fieldMgr.get_field<double>(
    avgProduction_->field_of_state(stk::mesh::StateNP1)
      .mesh_meta_data_ordinal());
  auto& avgTkeResN = fieldMgr.get_field<double>(
    avgTkeResolved_->field_of_state(stk::mesh::StateN)
      .mesh_meta_data_ordinal());
  auto& avgTkeResNp1 = fieldMgr.get_field<double>(
    avgTkeResolved_->field_of_state(stk::mesh::StateNP1)
      .mesh_meta_data_ordinal());
  auto& avgResAdeqN = fieldMgr.get_field<double>(
    avgResAdequacy_->field_of_state(stk::mesh::StateN)
      .mesh_meta_data_ordinal());
  auto& avgResAdeqNp1 = fieldMgr.get_field<double>(
    avgResAdequacy_->field_of_state(stk::mesh::StateNP1)
      .mesh_meta_data_ordinal());

  avgVelN.sync_to_device();
  avgDudxN.sync_to_device();
  avgProdN.sync_to_device();
  avgTkeResN.sync_to_device();
  avgResAdeqN.sync_to_device();

  const auto& meta = realm_.meta_data();
  const stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part() |
     meta.aura_part()) &
    stk::mesh::selectField(*avgVelocity_);
  nalu_ngp::field_copy(
    ngpMesh, sel, avgVelNp1, avgVelN, meta.spatial_dimension());
  nalu_ngp::field_copy(
    ngpMesh, sel, avgDudxNp1, avgDudxN,
    meta.spatial_dimension() * meta.spatial_dimension());
  nalu_ngp::field_copy(ngpMesh, sel, avgProdNp1, avgProdN, 1);
  nalu_ngp::field_copy(ngpMesh, sel, avgTkeResNp1, avgTkeResN, 1);
  nalu_ngp::field_copy(ngpMesh, sel, avgResAdeqNp1, avgResAdeqN, 1);
  avgVelNp1.modify_on_device();
  avgDudxNp1.modify_on_device();
  avgProdNp1.modify_on_device();
  avgTkeResNp1.modify_on_device();
  avgResAdeqNp1.modify_on_device();
}

} // namespace nalu
} // namespace sierra
