// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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
  massVofBalancedFlowRate_ = get_field_ordinal(
    meta, "mass_vof_balanced_flow_rate", stk::topology::EDGE_RANK);
  density_ =
    get_field_ordinal(realm.meta_data(), "density", stk::mesh::StateNP1);

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

void
VOFAdvectionEdgeAlg::execute()
{
  const double eps = 1.0e-11;
  const double gradient_eps = 1.0e-9;

  const int ndim = realm_.meta_data().spatial_dimension();
  const auto& meta = realm_.meta_data();

  const DblType alphaUpw = realm_.get_alpha_upw_factor("volume_of_fluid");
  const DblType hoUpwind = realm_.get_upw_factor("volume_of_fluid");
  const DblType relaxFac =
    realm_.solutionOptions_->get_relaxation_factor("volume_of_fluid");
  const bool useLimiter = realm_.primitive_uses_limiter("volume_of_fluid");

  const DblType om_alphaUpw = 1.0 - alphaUpw;

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto scalarQ = fieldMgr.get_field<double>(scalarQ_);
  const auto dqdx = fieldMgr.get_field<double>(dqdx_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto massFlowRate = fieldMgr.get_field<double>(massFlowRate_);
  const auto massVofBalancedFlowRate =
    fieldMgr.get_field<double>(massVofBalancedFlowRate_);
  const auto density = fieldMgr.get_field<double>(density_);

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

      const DblType mdot = massFlowRate.get(edge, 0);

      NALU_ALIGNED DblType densityL = density.get(nodeL, 0);
      NALU_ALIGNED DblType densityR = density.get(nodeR, 0);

      NALU_ALIGNED DblType rhoIp = 0.5 * (densityL + densityR);

      const DblType vdot = mdot / rhoIp;
      const DblType qNp1L = scalarQ.get(nodeL, 0);
      const DblType qNp1R = scalarQ.get(nodeR, 0);

      // Compute extrapolated dq/dx
      NALU_ALIGNED DblType dqL = 0.0;
      NALU_ALIGNED DblType dqR = 0.0;

      for (int j = 0; j < ndim; ++j) {
        const DblType dxj =
          0.5 * (coordinates.get(nodeR, j) - coordinates.get(nodeL, j));
        dqL += dxj * dqdx.get(nodeL, j);
        dqR += dxj * dqdx.get(nodeR, j);
      }

      NALU_ALIGNED DblType limitL = 1.0;
      NALU_ALIGNED DblType limitR = 1.0;

      if (useLimiter) {
        const auto dq = scalarQ.get(nodeR, 0) - scalarQ.get(nodeL, 0);
        const auto dqML = 4.0 * dqL - dq;
        const auto dqMR = 4.0 * dqR - dq;
        limitL = van_leer(dqML, dq, eps);
        limitR = van_leer(dqMR, dq, eps);
      }

      // Upwind extrapolation with limiter terms
      NALU_ALIGNED DblType qIpL;
      NALU_ALIGNED DblType qIpR;
      qIpL = scalarQ.get(nodeL, 0) + dqL * hoUpwind * limitL;
      qIpR = scalarQ.get(nodeR, 0) - dqR * hoUpwind * limitR;

      // Advective flux
      const DblType qIp = 0.5 * (qNp1R + qNp1L); // 2nd order central term
      // Upwinded term
      const DblType qUpw = (vdot > 0) ? (alphaUpw * qIpL + om_alphaUpw * qIp)
                                      : (alphaUpw * qIpR + om_alphaUpw * qIp);

      const DblType adv_flux = vdot * qUpw;
      smdata.rhs(0) -= adv_flux;
      smdata.rhs(1) += adv_flux;

      // Left node contribution; upwind terms
      DblType alhsfac = 0.5 * (vdot + stk::math::abs(vdot)) * alphaUpw;
      smdata.lhs(0, 0) += alhsfac / relaxFac;
      smdata.lhs(1, 0) -= alhsfac;

      // Right node contribution; upwind terms
      alhsfac = 0.5 * (vdot - stk::math::abs(vdot)) * alphaUpw;
      smdata.lhs(1, 1) -= alhsfac / relaxFac;
      smdata.lhs(0, 1) += alhsfac;

      // Compression term

      DblType dOmegadxMag = 0.0;
      DblType interface_gradient[3] = {0.0, 0.0, 0.0};

      for (int j = 0; j < ndim; ++j) {
        interface_gradient[j] = 0.5 * (dqdx.get(nodeL, j) + dqdx.get(nodeR, j));
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

      // Hard-coded values comes from Jain, 2022 to enforce
      // VOF function bounds of [0,1] while maintaining interface
      // thickness that is ~2 cells

      const DblType velocity_scale =
        5.0 * stk::math::abs(
                vdot /
                stk::math::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]));

      diffusion_coef = stk::math::sqrt(diffusion_coef) * 0.3;

      const DblType inv_axdx = 1.0 / axdx;

      const DblType dlhsfac = -velocity_scale * diffusion_coef * asq * inv_axdx;

      smdata.rhs(0) -= dlhsfac * (qNp1R - qNp1L);
      smdata.rhs(1) += dlhsfac * (qNp1R - qNp1L);

      massVofBalancedFlowRate.get(edge, 0) =
        dlhsfac * (qNp1R - qNp1L) * (density_liquid_ - density_gas_);

      smdata.lhs(0, 0) -= dlhsfac;
      smdata.lhs(0, 1) += dlhsfac;

      smdata.lhs(1, 0) += dlhsfac;
      smdata.lhs(1, 1) -= dlhsfac;

      const DblType omegaL =
        diffusion_coef * stk::math::log((qNp1L + eps) / (1.0 - qNp1L + eps));
      const DblType omegaR =
        diffusion_coef * stk::math::log((qNp1R + eps) / (1.0 - qNp1R + eps));
      const DblType omegaIp = 0.5 * (omegaL + omegaR);

      dOmegadxMag = 0.0;

      for (int d = 0; d < 3; ++d) {
        interface_gradient[d] = 0.0;
        interface_normal[d] = 0.0;
      }

      for (int j = 0; j < ndim; ++j) {
        interface_gradient[j] = 0.5 * (dqdx.get(nodeL, j) + dqdx.get(nodeR, j));
        interface_gradient[j] *= (2.0 * diffusion_coef * eps + diffusion_coef) /
                                 (eps * eps + eps - qIp * qIp + qIp);
      }

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

      massVofBalancedFlowRate.get(edge, 0) +=
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
