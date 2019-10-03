/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "TAMSAlgDriver.h"
#include "master_element/MasterElementFactory.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "ngp_utils/NgpTypes.h"
#include "ngp_utils/NgpFieldBLAS.h"
#include "ngp_algorithms/MetricTensorElemAlg.h"

namespace sierra {
namespace nalu {

class Realm;

TAMSAlgDriver::TAMSAlgDriver(Realm& realm)
  : realm_(realm),
    avgVelocity_(NULL),
    avgVelocityRTM_(NULL),
    avgTkeResolved_(NULL),
    avgDudx_(NULL),
    metric_(NULL),
    alpha_(NULL),
    resAdequacy_(NULL),
    avgResAdequacy_(NULL),
    avgProduction_(NULL),
    avgTime_(NULL),
    avgMdotScs_(NULL),
    avgMdot_(NULL),
    isInit_(true),
    metricTensorAlgDriver_(realm_, "metric_tensor"),
    avgMdotAlg_(realm_),
    turbulenceModel_(realm_.solutionOptions_->turbulenceModel_),
    resetTAMSAverages_(realm_.solutionOptions_->resetTAMSAverages_)
{
  if (turbulenceModel_ != SST_TAMS) {
    throw std::runtime_error(
      "User has requested TAMS, however, turbulence model has not been set "
      "to sst_tams, the only one supported by this equation system currently.");
  }
}

void
TAMSAlgDriver::register_fields_and_algorithms(
  stk::mesh::Part* part, const stk::topology& theTopo)
{
  stk::mesh::MetaData& meta = realm_.meta_data();
  const int nDim = meta.spatial_dimension();

  // Nodal fields
  alpha_ =
    &(meta.declare_field<ScalarFieldType>(stk::topology::NODE_RANK, "k_ratio"));
  stk::mesh::put_field_on_mesh(*alpha_, *part, nullptr);

  avgVelocity_ = &(meta.declare_field<VectorFieldType>(
    stk::topology::NODE_RANK, "average_velocity"));
  stk::mesh::put_field_on_mesh(*avgVelocity_, *part, nDim, nullptr);
  realm_.augment_restart_variable_list("average_velocity");

  if (
    realm_.solutionOptions_->meshMotion_ ||
    realm_.solutionOptions_->externalMeshDeformation_) {
    avgVelocityRTM_ = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "average_velocity_rtm"));
    stk::mesh::put_field_on_mesh(*avgVelocityRTM_, *part, nDim, nullptr);
    realm_.augment_restart_variable_list("average_velocity_rtm");
  }

  avgProduction_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_production"));
  stk::mesh::put_field_on_mesh(*avgProduction_, *part, nullptr);
  realm_.augment_restart_variable_list("average_production");

  avgDudx_ = &(meta.declare_field<GenericFieldType>(
    stk::topology::NODE_RANK, "average_dudx"));
  stk::mesh::put_field_on_mesh(*avgDudx_, *part, nDim * nDim, nullptr);
  realm_.augment_restart_variable_list("average_dudx");

  avgTkeResolved_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "average_tke_resolved"));
  stk::mesh::put_field_on_mesh(*avgTkeResolved_, *part, nullptr);
  realm_.augment_restart_variable_list("average_tke_resolved");

  avgTime_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "rans_time_scale"));
  stk::mesh::put_field_on_mesh(*avgTime_, *part, nullptr);

  metric_ = &(meta.declare_field<GenericFieldType>(
    stk::topology::NODE_RANK, "metric_tensor"));
  stk::mesh::put_field_on_mesh(*metric_, *part, nDim * nDim, nullptr);

  resAdequacy_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "resolution_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*resAdequacy_, *part, nullptr);

  avgResAdequacy_ = &(meta.declare_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "avg_res_adequacy_parameter"));
  stk::mesh::put_field_on_mesh(*avgResAdequacy_, *part, nullptr);
  realm_.augment_restart_variable_list("avg_res_adequacy_parameter");

  // Mdot fields
  if (realm_.realmUsesEdges_) {
    NaluEnv::self().naluOutputP0()
      << "Edge Mdot average added in TAMS " << std::endl;
    avgMdot_ = &(meta.declare_field<ScalarFieldType>(
      stk::topology::EDGE_RANK, "average_mass_flow_rate"));
    stk::mesh::put_field_on_mesh(*avgMdot_, *part, nullptr);
    realm_.augment_restart_variable_list("average_mass_flow_rate");
  } else {
    NaluEnv::self().naluOutputP0()
      << "Elemental Mdot average added in TAMS " << std::endl;
    MasterElement* meSCS =
      sierra::nalu::MasterElementRepo::get_surface_master_element(theTopo);
    const int numScsIp = meSCS->num_integration_points();

    avgMdotScs_ = &(meta.declare_field<GenericFieldType>(
      stk::topology::ELEMENT_RANK, "average_mass_flow_rate_scs"));
    stk::mesh::put_field_on_mesh(*avgMdotScs_, *part, numScsIp, nullptr);
    realm_.augment_restart_variable_list("average_mass_flow_rate_scs");
  }

  // Algorithms
  const AlgorithmType algType = INTERIOR;

  metricTensorAlgDriver_.register_elem_algorithm<MetricTensorElemAlg>(
    algType, part, "metric_tensor");

  if (!avgAlg_) {
    switch (realm_.solutionOptions_->turbulenceModel_) {
    case SST_TAMS:
      avgAlg_.reset(new SSTTAMSAveragesAlg(realm_, part));
      break;
    default:
      throw std::runtime_error("TAMSEquationSystem: non-supported turb model");
    }
  } else {
    avgAlg_->partVec_.push_back(part);
  }

  // avgMdot algorithm
  if (realm_.realmUsesEdges_) {
    avgMdotAlg_.register_edge_algorithm<TAMSAvgMdotEdgeAlg>(
      algType, part, "tams_avg_mdot_edge");
  } else {
    avgMdotAlg_.register_elem_algorithm<TAMSAvgMdotElemAlg>(
      algType, part, "tams_avg_mdot_elem");
  }

  // FIXME: tvisc needed for TAMS update, but is updated in LowMach...
  //        Perhaps there is a way to call tvisc from LowMach here?
  if (!tviscAlg_) {
    switch (realm_.solutionOptions_->turbulenceModel_) {
    case SST_TAMS:
      tviscAlg_.reset(new TurbViscSSTAlg(realm_, part, true));
      break;
    default:
      throw std::runtime_error("non-supported turb model in TAMS Eq Sys");
    }
  } else {
    tviscAlg_->partVec_.push_back(part);
  }
}

void
TAMSAlgDriver::initial_work()
{
  using Traits = nalu_ngp::NGPMeshTraits<ngp::Mesh>;

  compute_metric_tensor();

  // Initialize average_velocity, avg_dudx, avg_Prod
  // We don't want to do this on restart where TAMS fields are present
  if (resetTAMSAverages_) {
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

    // Need to update tvisc (avgDudx didn't exist)
    // before this to compute production
    tviscAlg_->execute();
    const auto tvisc = fieldMgr.get_field<double>(
      get_field_ordinal(meta, "turbulent_viscosity"));

    // Compute average production
    auto avgProd =
      fieldMgr.get_field<double>(avgProduction_->mesh_meta_data_ordinal());
    nalu_ngp::run_entity_algorithm(
      ngpMesh, stk::topology::NODE_RANK, sel,
      KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
        std::vector<DblType> tij(nDim * nDim, 0.0);
        for (int i = 0; i < nDim; ++i) {
          for (int j = 0; j < nDim; ++j) {
            const DblType avgSij = 0.5 * (avgDudx.get(mi, i * nDim + j) +
                                          avgDudx.get(mi, j * nDim + i));
            tij[i * nDim + j] = 2.0 * tvisc.get(mi, 0) * avgSij;
          }
        }

        std::vector<DblType> Pij(nDim * nDim, 0.0);
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

  //FIXME initialize mdot here
}

void
TAMSAlgDriver::execute()
{
  if (
    realm_.solutionOptions_->meshMotion_ ||
    realm_.solutionOptions_->externalMeshDeformation_) {
    metricTensorAlgDriver_.execute();
  }
  avgAlg_->execute();
  if (
    realm_.solutionOptions_->meshMotion_ ||
    realm_.solutionOptions_->externalMeshDeformation_) {
    realm_.compute_vrtm("average_velocity");
  }
  avgMdotAlg_.execute();
}

} // namespace nalu
} // namespace sierra
