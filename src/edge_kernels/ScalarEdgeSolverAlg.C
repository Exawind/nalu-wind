// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/ScalarEdgeSolverAlg.h"
#include "EquationSystem.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "edge_kernels/EdgeKernelUtils.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

ScalarEdgeSolverAlg::ScalarEdgeSolverAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  ScalarFieldType* scalarQ,
  VectorFieldType* dqdx,
  ScalarFieldType* diffFluxCoeff,
  const bool useAverages)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem),
    dofName_(scalarQ->name())
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  const std::string vrtmName =
    realm.does_mesh_move() ? "velocity_rtm" : "velocity";
  const std::string avgVrtmName =
    realm.does_mesh_move() ? "average_velocity_rtm" : "average_velocity";

  scalarQ_ = scalarQ->mesh_meta_data_ordinal();
  dqdx_ = dqdx->mesh_meta_data_ordinal();
  diffFluxCoeff_ = diffFluxCoeff->mesh_meta_data_ordinal();
  density_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  massFlowRate_ = get_field_ordinal(
    meta, (useAverages) ? "average_mass_flow_rate" : "mass_flow_rate",
    stk::topology::EDGE_RANK);
  velocityRTM_ =
    get_field_ordinal(meta, (useAverages) ? avgVrtmName : vrtmName);
  pecletFunction_ = eqSystem->ngp_create_peclet_function<double>(dofName_);
}

void
ScalarEdgeSolverAlg::execute()
{
  const double eps = 1.0e-16;
  const int ndim = realm_.meta_data().spatial_dimension();

  const DblType alpha = realm_.get_alpha_factor(dofName_);
  const DblType alphaUpw = realm_.get_alpha_upw_factor(dofName_);
  const DblType hoUpwind = realm_.get_upw_factor(dofName_);
  const DblType relaxFac =
    realm_.solutionOptions_->get_relaxation_factor(dofName_);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName_);

  const DblType om_alpha = 1.0 - alpha;
  const DblType om_alphaUpw = 1.0 - alphaUpw;

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto vrtm = fieldMgr.get_field<double>(velocityRTM_);
  const auto scalarQ = fieldMgr.get_field<double>(scalarQ_);
  const auto dqdx = fieldMgr.get_field<double>(dqdx_);
  const auto density = fieldMgr.get_field<double>(density_);
  const auto dflux = fieldMgr.get_field<double>(diffFluxCoeff_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto massFlowRate = fieldMgr.get_field<double>(massFlowRate_);

  // Local pointer for device capture
  auto* pecFunc = pecletFunction_;

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType & smdata, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Scratch work array for edgeAreaVector
      DblType av[NDimMax_];
      // Populate area vector work array
      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      const DblType mdot = massFlowRate.get(edge, 0);

      const DblType densityL = density.get(nodeL, 0);
      const DblType densityR = density.get(nodeR, 0);

      const DblType qNp1L = scalarQ.get(nodeL, 0);
      const DblType qNp1R = scalarQ.get(nodeR, 0);

      const DblType viscosityL = dflux.get(nodeL, 0);
      const DblType viscosityR = dflux.get(nodeR, 0);

      const DblType viscIp = 0.5 * (viscosityL + viscosityR);
      const DblType diffIp =
        0.5 * (viscosityL / densityL + viscosityR / densityR);

      // Compute area vector related quantities and (U dot areaVec)
      DblType axdx = 0.0;
      DblType asq = 0.0;
      DblType udotx = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
        udotx += 0.5 * dxj * (vrtm.get(nodeR, d) + vrtm.get(nodeL, d));
      }
      const DblType inv_axdx = 1.0 / axdx;

      // Compute extrapolated dq/dx
      DblType dqL = 0.0;
      DblType dqR = 0.0;
      DblType nonOrth = 0.0;

      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          (coordinates.get(nodeR, d) - coordinates.get(nodeL, d));
        dqL += 0.5 * dxj * dqdx.get(nodeL, d);
        dqR += 0.5 * dxj * dqdx.get(nodeR, d);

        const DblType kxj = av[d] - asq * inv_axdx * dxj;
        nonOrth +=
          -viscIp * kxj * 0.5 * (dqdx.get(nodeR, d) + dqdx.get(nodeL, d));
      }

      const DblType pecnum = stk::math::abs(udotx) / (diffIp + eps);
      const DblType pecfac = pecFunc->execute(pecnum);
      const DblType om_pecfac = 1.0 - pecfac;

      DblType limitL = 1.0;
      DblType limitR = 1.0;
      if (useLimiter) {
        const auto dq = qNp1R - qNp1L;
        const auto dqML = 4.0 * dqL - dq;
        const auto dqMR = 4.0 * dqR - dq;
        limitL = van_leer(dqML, dq, eps);
        limitR = van_leer(dqMR, dq, eps);
      }

      const DblType qIpL = qNp1L + dqL * hoUpwind * limitL;
      const DblType qIpR = qNp1R - dqR * hoUpwind * limitR;

      // Diffusive flux
      const DblType lhsfac = -viscIp * asq * inv_axdx;
      const DblType diffFlux = lhsfac * (qNp1R - qNp1L) + nonOrth;

      // Left node
      smdata.lhs(0, 0) = -lhsfac / relaxFac;
      smdata.lhs(0, 1) = lhsfac;
      smdata.rhs(0) = -diffFlux;
      // Right node
      smdata.lhs(1, 0) = lhsfac;
      smdata.lhs(1, 1) = -lhsfac / relaxFac;
      smdata.rhs(1) = diffFlux;

      // Advective flux
      const DblType qIp = 0.5 * (qNp1R + qNp1L); // 2nd order central term

      // Upwinded term
      const DblType qUpw = (mdot > 0) ? (alphaUpw * qIpL + om_alphaUpw * qIp)
                                      : (alphaUpw * qIpR + om_alphaUpw * qIp);

      const DblType qHatL = (alpha * qIpL + om_alpha * qIp);
      const DblType qHatR = (alpha * qIpR + om_alpha * qIp);
      const DblType qCds = 0.5 * (qHatL + qHatR);

      const DblType adv_flux = mdot * (pecfac * qUpw + om_pecfac * qCds);
      smdata.rhs(0) -= adv_flux;
      smdata.rhs(1) += adv_flux;

      // Left node contribution; upwind terms
      DblType alhsfac =
        0.5 * (mdot + stk::math::abs(mdot)) * pecfac * alphaUpw +
        0.5 * alpha * om_pecfac * mdot;
      smdata.lhs(0, 0) += alhsfac / relaxFac;
      smdata.lhs(1, 0) -= alhsfac;

      // Right node contribution; upwind terms
      alhsfac = 0.5 * (mdot - stk::math::abs(mdot)) * pecfac * alphaUpw +
                0.5 * alpha * om_pecfac * mdot;
      smdata.lhs(1, 1) -= alhsfac / relaxFac;
      smdata.lhs(0, 1) += alhsfac;

      // central terms
      alhsfac = 0.5 * mdot * (pecfac * om_alphaUpw + om_pecfac * om_alpha);
      smdata.lhs(0, 0) += alhsfac / relaxFac;
      smdata.lhs(0, 1) += alhsfac;
      smdata.lhs(1, 0) -= alhsfac;
      smdata.lhs(1, 1) -= alhsfac / relaxFac;
    });
}

} // namespace nalu
} // namespace sierra
