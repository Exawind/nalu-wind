// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include "NaluEnv.h"
#include "edge_kernels/VOFAdvectionEdgeAlg.h"
#include "EquationSystem.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "edge_kernels/EdgeKernelUtils.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"
#include <stk_math/StkMath.hpp>
#include <property_evaluator/MaterialPropertyData.h>
#include <stk_util/parallel/ParallelReduce.hpp>
#include "ngp_utils/NgpLoopUtils.h"

namespace sierra {
namespace nalu {

VOFAdvectionEdgeAlg::VOFAdvectionEdgeAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  ScalarFieldType* scalarQ,
  VectorFieldType* dqdx,
  const bool useAverages)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  scalarQ_ = scalarQ->mesh_meta_data_ordinal();
  dqdx_ = dqdx->mesh_meta_data_ordinal();
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  massFlowRate_ = get_field_ordinal(
    meta, (useAverages) ? "average_mass_flow_rate" : "mass_flow_rate",
    stk::topology::EDGE_RANK);
  massForcedFlowRate_ =
    get_field_ordinal(meta, "mass_forced_flow_rate", stk::topology::EDGE_RANK);
  density_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  velocity_n_ = get_field_ordinal(meta, "velocity", stk::mesh::StateN);

  std::map<PropertyIdentifier, MaterialPropertyData*>::iterator itf =
    realm_.materialPropertys_.propertyDataMap_.find(DENSITY_ID);

  // Hard set value here for unit testing without property map.
  if (itf == realm_.materialPropertys_.propertyDataMap_.end()) {
    density_liquid_ = 1000.0;
    density_gas_ = 1.0;
  } else {
    auto mdata = (*itf).second;

    switch (mdata->type_) {
    case CONSTANT_MAT: {
      density_liquid_ = mdata->constValue_;
      density_gas_ = mdata->constValue_;
      break;
    }
    case VOF_MAT: {
      density_liquid_ = mdata->primary_;
      density_gas_ = mdata->secondary_;
      break;
    }
    default: {
      throw std::runtime_error("Incorrect density property set for VOF "
                               "calculations. Use a constant or "
                               "VOF property for density.");
      break;
    }
    }
  }
}
// The following scheme is a modified form of Jain, 2022 VOF paper.
// The sharpening term is taken directly from the paper.
// The forcing terms used in the momentum equation from the paper are
// calculated as a new mass flux that moves fluids according to the
// sharpening/diffusion of the VOF function.
// qIp and rhoIp are based on the resulting drho/dt of the scheme
// to ensure exact consistency with momentum and continuity.
// Note that upwinding of VOF is not necessary because there is
// a diffusion term for VOF that provides sufficient dissipation.
void
VOFAdvectionEdgeAlg::execute()
{
  const double eps = 1.0e-11;
  const double gradient_eps = 1.0e-9;

  const int ndim = realm_.meta_data().spatial_dimension();

  const DblType relaxFac =
    realm_.solutionOptions_->get_relaxation_factor("volume_of_fluid");

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto scalarQ = fieldMgr.get_field<double>(scalarQ_);
  const auto dqdx = fieldMgr.get_field<double>(dqdx_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto massFlowRate = fieldMgr.get_field<double>(massFlowRate_);
  const auto massForcedFlowRate =
    fieldMgr.get_field<double>(massForcedFlowRate_);
  const auto density = fieldMgr.get_field<double>(density_);
  const auto velocity_n = fieldMgr.get_field<double>(velocity_n_);

  stk::mesh::MetaData& meta_data = realm_.meta_data();

  auto velocity_field_ =
    meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");

  const stk::mesh::Selector sel =
    (meta_data.locally_owned_part() | meta_data.globally_shared_part()) &
    stk::mesh::selectField(*velocity_field_);

  double local_max_velocity = 0.0;
  double global_max_velocity = 0.0;

  const std::string algName = "calc_velocity_scale_vof";

  const auto& ngpMesh = realm_.ngp_mesh();

  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  Kokkos::Max<double> max_velocity_reducer(local_max_velocity);
  nalu_ngp::run_entity_par_reduce(
    algName, ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi, double& pSum) {
      double vel_squared = 0.0;
      for (int idim = 0; idim < ndim; ++idim)
        vel_squared += velocity_n.get(mi, idim) * velocity_n.get(mi, idim);
      pSum = stk::math::max(stk::math::sqrt(vel_squared), pSum);
    },
    max_velocity_reducer);
  stk::all_reduce_max(
    NaluEnv::self().parallel_comm(), &local_max_velocity, &global_max_velocity,
    1);
  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType & smdata, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Scratch work array for edgeAreaVector
      NALU_ALIGNED DblType av[NDimMax_];

      // Populate area vector work array
      for (int d = 0; d < ndim; ++d) {
        av[d] = edgeAreaVec.get(edge, d);
      }

      NALU_ALIGNED DblType densityL = density.get(nodeL, 0);
      NALU_ALIGNED DblType densityR = density.get(nodeR, 0);
      const DblType rhoIp = 0.5 * densityL + 0.5 * densityR;

      const DblType vdot = massFlowRate.get(edge, 0) / rhoIp;

      const DblType qNp1L = scalarQ.get(nodeL, 0);
      const DblType qNp1R = scalarQ.get(nodeR, 0);

      // Advective flux
      const DblType qIp = 0.5 * (qNp1R + qNp1L);

      const DblType adv_flux = vdot * qIp;
      smdata.rhs(0) -= adv_flux;
      smdata.rhs(1) += adv_flux;

      DblType alhsfac = 0.5 * vdot;
      smdata.lhs(0, 0) += alhsfac / relaxFac;
      smdata.lhs(1, 0) -= alhsfac;

      alhsfac = 0.5 * vdot;
      smdata.lhs(1, 1) -= alhsfac / relaxFac;
      smdata.lhs(0, 1) += alhsfac;

      // Compression + Diffusion term
      // Hard coded 5.0 value comes from Jain, 2022 based on
      // enforcing bounds of VOF function to [0,1] while maintaining
      // interface that is approx ~2 cells thick.
      const DblType velocity_scale = global_max_velocity * 5.0;

      DblType axdx = 0.0;
      DblType asq = 0.0;
      DblType diffusion_coef = 0.0;

      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        diffusion_coef += dxj * dxj;
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
      }

      // Hard-coded 0.6 value comes from Jain, 2022 to enforce
      // VOF function bounds of [0,1] while maintaining interface
      // thickness that is ~2 cells
      diffusion_coef = stk::math::sqrt(diffusion_coef) * 0.6;

      const DblType inv_axdx = 1.0 / axdx;

      const DblType dlhsfac = -velocity_scale * diffusion_coef * asq * inv_axdx;

      smdata.rhs(0) -= dlhsfac * (qNp1R - qNp1L);
      smdata.rhs(1) += dlhsfac * (qNp1R - qNp1L);

      massForcedFlowRate.get(edge, 0) =
        dlhsfac * (qNp1R - qNp1L) * (density_liquid_ - density_gas_);

      smdata.lhs(0, 0) -= dlhsfac;
      smdata.lhs(0, 1) += dlhsfac;

      smdata.lhs(1, 0) += dlhsfac;
      smdata.lhs(1, 1) -= dlhsfac;

      DblType dOmegadxMag = 0.0;

      const DblType omegaL =
        diffusion_coef * stk::math::log((qNp1L + eps) / (1.0 - qNp1L + eps));
      const DblType omegaR =
        diffusion_coef * stk::math::log((qNp1R + eps) / (1.0 - qNp1R + eps));
      const DblType omegaIp = 0.5 * (omegaL + omegaR);
      DblType interface_gradient[3] = {0.0, 0.0, 0.0};

      for (int j = 0; j < ndim; ++j) {
        interface_gradient[j] = 0.5 * (dqdx.get(nodeL, j) + dqdx.get(nodeR, j));
        interface_gradient[j] *= (2.0 * diffusion_coef * eps + diffusion_coef) /
                                 (eps * eps + eps - qIp * qIp + qIp);
      }

      DblType interface_normal[3] = {0.0, 0.0, 0.0};

      for (int j = 0; j < ndim; ++j)
        dOmegadxMag += interface_gradient[j] * interface_gradient[j];

      dOmegadxMag = stk::math::sqrt(dOmegadxMag);

      // No gradient == no interface
      if (dOmegadxMag < gradient_eps)
        return;

      for (int d = 0; d < ndim; ++d)
        interface_normal[d] = interface_gradient[d] / dOmegadxMag;

      DblType compression = 0.0;

      for (int d = 0; d < ndim; ++d)
        compression +=
          velocity_scale * 0.25 *
          (1.0 - stk::math::tanh(0.5 * omegaIp / diffusion_coef) *
                   stk::math::tanh(0.5 * omegaIp / diffusion_coef)) *
          interface_normal[d] * av[d];

      smdata.rhs(0) -= compression;
      smdata.rhs(1) += compression;

      massForcedFlowRate.get(edge, 0) +=
        compression * (density_liquid_ - density_gas_);

      // Left node contribution; Lag in iterations except for central 0.5*q term
      DblType slhsfac = 0.0;
      for (int d = 0; d < ndim; ++d)
        slhsfac += 0.5 * interface_normal[d] * 1.5 * velocity_scale *
                   (1.0 - qIp) * av[d];

      smdata.lhs(0, 0) += slhsfac / relaxFac;
      smdata.lhs(1, 0) -= slhsfac;

      // Right node contribution;
      smdata.lhs(1, 1) -= slhsfac / relaxFac;
      smdata.lhs(0, 1) += slhsfac;
    });
}

} // namespace nalu
} // namespace sierra
