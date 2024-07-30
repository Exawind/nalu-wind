// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/MomentumEdgeSolverAlg.h"
#include "EquationSystem.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "edge_kernels/EdgeKernelUtils.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

MomentumEdgeSolverAlg::MomentumEdgeSolverAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  const std::string vrtmName =
    realm.does_mesh_move() ? "velocity_rtm" : "velocity";
  velocityRTM_ = get_field_ordinal(meta, vrtmName);

  const std::string velName = "velocity";
  velocity_ = get_field_ordinal(meta, velName, stk::mesh::StateNP1);

  std::string viscName;
  if (
    (realm.is_turbulent()) &&
    (realm.solutionOptions_->turbulenceModel_ != TurbulenceModel::SST_AMS))
    viscName = "effective_viscosity_u";
  else
    viscName = "viscosity";

  density_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  viscosity_ = get_field_ordinal(meta, viscName);
  dudx_ = get_field_ordinal(meta, "dudx");
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  massFlowRate_ =
    get_field_ordinal(meta, "mass_flow_rate", stk::topology::EDGE_RANK);
  massVofBalancedFlowRate_ =
    realm.solutionOptions_->realm_has_vof_
      ? get_field_ordinal(
          meta, "mass_vof_balanced_flow_rate", stk::topology::EDGE_RANK)
      : massFlowRate_;
  pecletFactor_ =
    get_field_ordinal(meta, "peclet_factor", stk::topology::EDGE_RANK);
  maskNodeField_ = get_field_ordinal(
    meta, "abl_wall_no_slip_wall_func_node_mask", stk::topology::NODE_RANK);

  if (realm_.solutionOptions_->realm_has_vof_) {
    NaluEnv::self().naluOutputP0()
      << "WARNING: volume_of_fluid is present. For stability, upwinding in the "
         "MomentumEdgeSolverAlg is automatically turned on near the liquid-gas "
         "interface.\n";
  }
}

void
MomentumEdgeSolverAlg::execute()
{
  const double eps = 1.0e-16;
  const int ndim = realm_.meta_data().spatial_dimension();

  const std::string dofName = "velocity";
  const DblType includeDivU = realm_.get_divU();
  const DblType alpha = realm_.get_alpha_factor(dofName);
  const DblType alphaUpw_input = realm_.get_alpha_upw_factor(dofName);
  const DblType hoUpwind = realm_.get_upw_factor(dofName);
  const DblType relaxFacU =
    realm_.solutionOptions_->get_relaxation_factor(dofName);
  const bool useLimiter = realm_.primitive_uses_limiter(dofName);

  const DblType om_alpha = 1.0 - alpha;
  const DblType om_alphaUpw_input = 1.0 - alphaUpw_input;

  const DblType has_vof = (double)realm_.solutionOptions_->realm_has_vof_;

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto vrtm = fieldMgr.get_field<double>(velocityRTM_);
  const auto vel = fieldMgr.get_field<double>(velocity_);
  const auto dudx = fieldMgr.get_field<double>(dudx_);
  const auto viscosity = fieldMgr.get_field<double>(viscosity_);
  const auto density = fieldMgr.get_field<double>(density_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto massFlowRate = fieldMgr.get_field<double>(massFlowRate_);
  const auto massVofBalancedFlowRate =
    fieldMgr.get_field<double>(massVofBalancedFlowRate_);
  const auto pecletFactor = fieldMgr.get_field<double>(pecletFactor_);
  const auto maskNodeField = fieldMgr.get_field<double>(maskNodeField_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType & smdata, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Variables are read-only when C++ captures by value into a lambda
      // This avoids this issue, allowing modification of upwinding parameters
      // for VOF
      DblType alphaUpw = alphaUpw_input;
      DblType om_alphaUpw = om_alphaUpw_input;

      // Scratch work array for edgeAreaVector
      NALU_ALIGNED DblType av[NDimMax_];
      // Populate area vector work array
      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      const DblType mdot =
        massFlowRate.get(edge, 0) + has_vof * massVofBalancedFlowRate(edge, 0);

      const DblType densityL = density.get(nodeL, 0);
      const DblType densityR = density.get(nodeR, 0);
      const DblType viscosityL = viscosity.get(nodeL, 0);
      const DblType viscosityR = viscosity.get(nodeR, 0);

      const DblType viscIp = 0.5 * (viscosityL + viscosityR);

      // Compute area vector related quantities and (U dot areaVec)
      DblType axdx = 0.0;
      DblType asq = 0.0;
      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += av[d] * av[d];
        axdx += av[d] * dxj;
      }
      const DblType inv_axdx = 1.0 / axdx;

      // Compute extrapolated du/dx
      NALU_ALIGNED DblType duL[NDimMax_];
      NALU_ALIGNED DblType duR[NDimMax_];

      for (int i = 0; i < ndim; ++i) {
        const int offset = i * ndim;
        duL[i] = 0.0;
        duR[i] = 0.0;

        for (int j = 0; j < ndim; ++j) {
          const DblType dxj =
            0.5 * (coordinates.get(nodeR, j) - coordinates.get(nodeL, j));
          duL[i] += dxj * dudx.get(nodeL, offset + j);
          duR[i] += dxj * dudx.get(nodeR, offset + j);
        }
      }

      NALU_ALIGNED DblType limitL[NDimMax_] = {1.0, 1.0, 1.0};
      NALU_ALIGNED DblType limitR[NDimMax_] = {1.0, 1.0, 1.0};

      if (useLimiter) {
        for (int d = 0; d < ndim; ++d) {
          const auto du = vel.get(nodeR, d) - vel.get(nodeL, d);
          const auto duML = 4.0 * duL[d] - du;
          const auto duMR = 4.0 * duR[d] - du;
          limitL[d] = van_leer(duML, du, eps);
          limitR[d] = van_leer(duMR, du, eps);
        }
      }

      // Upwinding switch for multiphase cases:
      // Factors determined by ensuring full upwinding at interfaces with
      // interface widths that are ~2 cells thick
      DblType pecfac = pecletFactor.get(edge, 0);
      DblType om_pecfac = 1.0 - pecfac;
      DblType density_upwinding_factor = 1.0;
      if (has_vof > 0.5) {
        const DblType min_density = stk::math::min(densityL, densityR);
        const DblType density_differential =
          stk::math::abs(densityL - densityR) / min_density;
        density_upwinding_factor =
          1.0 - stk::math::erf(6.0 * density_differential);

        alphaUpw = density_upwinding_factor * alphaUpw +
                   (1.0 - density_upwinding_factor);
        om_alphaUpw = 1.0 - alphaUpw;
        pecfac =
          1.0 - density_upwinding_factor + density_upwinding_factor * pecfac;
        om_pecfac = 1.0 - pecfac;
      }

      // Upwind extrapolation with limiter terms
      NALU_ALIGNED DblType uIpL[NDimMax_];
      NALU_ALIGNED DblType uIpR[NDimMax_];
      for (int d = 0; d < ndim; ++d) {
        uIpL[d] = vel.get(nodeL, d) +
                  duL[d] * hoUpwind * limitL[d] * density_upwinding_factor;
        uIpR[d] = vel.get(nodeR, d) -
                  duR[d] * hoUpwind * limitR[d] * density_upwinding_factor;
      }

      // TODO(psakiev) extract this a funciton into EdgeKernelUtils.h
      // Computation of duidxj term, reproduce original comment by S. P. Domino
      /*
        form duidxj with over-relaxed procedure of Jasak:

        dui/dxj = GjUi +[(uiR - uiL) - GlUi*dxl]*Aj/AxDx
        where Gp is the interpolated pth nodal gradient for ui
      */
      NALU_ALIGNED DblType duidxj[NDimMax_][NDimMax_];
      for (int i = 0; i < ndim; ++i) {
        const auto dui = vel.get(nodeR, i) - vel.get(nodeL, i);
        const auto offset = i * ndim;

        // Non-orthogonal correction
        DblType gjuidx = 0.0;
        for (int j = 0; j < ndim; ++j) {
          const DblType dxj =
            coordinates.get(nodeR, j) - coordinates.get(nodeL, j);
          const DblType gjui =
            0.5 * (dudx.get(nodeR, offset + j) + dudx.get(nodeL, offset + j));
          gjuidx += gjui * dxj;
        }

        // final dui/dxj with non-orthogonal correction contributions
        for (int j = 0; j < ndim; ++j) {
          const DblType gjui =
            0.5 * (dudx.get(nodeR, offset + j) + dudx.get(nodeL, offset + j));
          duidxj[i][j] = gjui + (dui - gjuidx) * av[j] * inv_axdx;
        }
      }

      // diffusion LHS term
      const DblType dlhsfac = -viscIp * asq * inv_axdx;

      for (int i = 0; i < ndim; ++i) {
        // Left and right row/col indices
        const int rowL = i;
        const int rowR = i + ndim;

        const DblType uiIp = 0.5 * (vel.get(nodeR, i) + vel.get(nodeL, i));

        // Upwind contribution
        const DblType uiUpw = (mdot > 0.0)
                                ? (alphaUpw * uIpL[i] + om_alphaUpw * uiIp)
                                : (alphaUpw * uIpR[i] + om_alphaUpw * uiIp);

        const DblType uiHatL = (alpha * uIpL[i] + om_alpha * uiIp);
        const DblType uiHatR = (alpha * uIpR[i] + om_alpha * uiIp);
        const DblType uiCds = 0.5 * (uiHatL + uiHatR);

        // Advective flux
        const DblType adv_flux = mdot * (pecfac * uiUpw + om_pecfac * uiCds);

        DblType diff_flux = 0.0;
        // div(U) part first
        for (int j = 0; j < ndim; ++j)
          diff_flux += duidxj[j][j];
        diff_flux *= 2.0 / 3.0 * viscIp * av[i] * includeDivU;

        for (int j = 0; j < ndim; ++j)
          diff_flux += -viscIp * (duidxj[i][j] + duidxj[j][i]) * av[j];

        const DblType maskNode = stk::math::min(
          maskNodeField.get(nodeL, 0), maskNodeField.get(nodeR, 0));
        const DblType total_flux = adv_flux + diff_flux * maskNode;

        smdata.rhs(rowL) -= total_flux;
        smdata.rhs(rowR) += total_flux;

        // Left node contribution; upwind terms
        DblType alhsfac =
          0.5 * (mdot + stk::math::abs(mdot)) * pecfac * alphaUpw +
          0.5 * alpha * om_pecfac * mdot;
        smdata.lhs(rowL, rowL) += alhsfac / relaxFacU;
        smdata.lhs(rowR, rowL) -= alhsfac;

        // Right node contribution; upwind terms
        alhsfac = 0.5 * (mdot - stk::math::abs(mdot)) * pecfac * alphaUpw +
                  0.5 * alpha * om_pecfac * mdot;
        smdata.lhs(rowR, rowR) -= alhsfac / relaxFacU;
        smdata.lhs(rowL, rowR) += alhsfac;

        // central terms
        alhsfac = 0.5 * mdot * (pecfac * om_alphaUpw + om_pecfac * om_alpha);
        smdata.lhs(rowL, rowL) += alhsfac / relaxFacU;
        smdata.lhs(rowL, rowR) += alhsfac;
        smdata.lhs(rowR, rowL) -= alhsfac;
        smdata.lhs(rowR, rowR) -= alhsfac / relaxFacU;

        // Diffusion terms
        smdata.lhs(rowL, rowL) -= dlhsfac / relaxFacU;
        smdata.lhs(rowL, rowR) += dlhsfac;
        smdata.lhs(rowR, rowL) += dlhsfac;
        smdata.lhs(rowR, rowR) -= dlhsfac / relaxFacU;

        for (int j = 0; j < ndim; ++j) {
          const DblType lhsfacNS = -viscIp * av[i] * av[j] * inv_axdx;

          const int colL = j;
          const int colR = j + ndim;

          smdata.lhs(rowL, colL) -= lhsfacNS / relaxFacU;
          smdata.lhs(rowL, colR) += lhsfacNS;
          smdata.lhs(rowR, colL) += lhsfacNS;
          smdata.lhs(rowR, colR) -= lhsfacNS / relaxFacU;
        }
      }
    });
}

} // namespace nalu
} // namespace sierra
