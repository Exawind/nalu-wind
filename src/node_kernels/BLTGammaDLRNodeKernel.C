/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level KynemaUGF */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/BLTGammaDLRNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace kynema_ugf {

BLTGammaDLRNodeKernel::BLTGammaDLRNodeKernel(
  const stk::mesh::MetaData& meta,
  const std::string& velocityName,
  const double uInf)
  : NGPNodeKernel<BLTGammaDLRNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    velocityID_(get_field_ordinal(meta, velocityName)),
    pressureID_(get_field_ordinal(meta, "pressure")),
    dpdxID_(get_field_ordinal(meta, "dpdx")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    uInf_(uInf),
    nDim_(meta.spatial_dimension())
{
}

void
BLTGammaDLRNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  velocity_ = fieldMgr.get_field<double>(velocityID_);
  pressure_ = fieldMgr.get_field<double>(pressureID_);
  dpdx_ = fieldMgr.get_field<double>(dpdxID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  minD_ = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  gamint_ = fieldMgr.get_field<double>(gamintID_);

  fsti_ = realm.get_turb_model_constant(TM_fsti);
}

KOKKOS_FUNCTION
void
BLTGammaDLRNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType small = 1.0e-16;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType gamint = gamint_.get(node, 0);

  const DblType density = density_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType minD = minD_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  // Gas constants of air at sea-level reference pressure
  const DblType pinf = 101325.0;
  const DblType Rair = 8314.4621 / 28.96;
  const DblType Tinf = pinf / (Rair * density);
  const DblType kappa = 1.4;
  const DblType km1 = kappa - 1.0;

  // pressure field is relative to the reference pressure
  const DblType pressure = pressure_.get(node, 0) + pinf;

  // constants for the source terms
  const DblType flength = 14.0;
  const DblType caTwo = 0.06;
  const DblType ceTwo = 50.0;

  const DblType Uinf = uInf_;
  const DblType Minf = Uinf / stk::math::sqrt(kappa * Rair * Tinf);

  DblType sijMag = 0.0;
  DblType vortMag = 0.0;

  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      const DblType duidxj = dudx_.get(node, nDim_ * i + j);
      const DblType dujdxi = dudx_.get(node, nDim_ * j + i);

      const DblType rateOfStrain = 0.5 * (duidxj + dujdxi);
      const DblType vortTensor = 0.5 * (duidxj - dujdxi);
      sijMag += rateOfStrain * rateOfStrain;
      vortMag += vortTensor * vortTensor;
    }
  }

  sijMag = stk::math::sqrt(2.0 * sijMag);
  vortMag = stk::math::sqrt(2.0 * vortMag);

  const DblType Rev = density * minD * minD * sijMag / visc;

  DblType TuL = 0.0;
  if (fsti_ > 0.0) {
    TuL = fsti_; // constant Tu from input
  } else {
    TuL = stk::math::min(
      100.0 * stk::math::sqrt(stk::math::max(2.0 / 3.0 * tke, 0.0)) /
        stk::math::max(sdr, small) / (minD + 1.0e-10),
      100.0); // local Tu
  }
  TuL = stk::math::max(TuL, small);

  // Boundary layer edge values from isentropic relations
  const DblType pressureRatio = stk::math::max(pressure / pinf, small);
  const DblType pRatioExp = stk::math::pow(pressureRatio, km1 / kappa);
  const DblType var1 = 1.0 + 0.5 * km1 * Minf * Minf;

  const DblType Me =
    stk::math::sqrt(stk::math::max((var1 / pRatioExp - 1.0) * 2.0 / km1, 0.0));
  const DblType rhoe = density * stk::math::pow(pressureRatio, 1.0 / kappa);
  const DblType Te = Tinf * pRatioExp;
  const DblType Ue = stk::math::sqrt(
    stk::math::max(
      Uinf * Uinf + 2.0 * kappa / km1 * (1.0 - pRatioExp) * pinf / density,
      small));

  // Sutherland's law referenced to freestream conditions
  const DblType Su = 110.4;
  const DblType Tref = Tinf;
  const DblType muref = visc;

  const DblType nue =
    muref / rhoe * stk::math::pow(Te / Tref, 1.5) * (Tref + Su) / (Te + Su);

  // Streamwise edge-velocity gradient: dUe/ds = (u/|u|) . grad(Ue)
  DblType uMag = 0.0;
  for (int i = 0; i < nDim_; ++i)
    uMag += velocity_.get(node, i) * velocity_.get(node, i);
  uMag = stk::math::max(stk::math::sqrt(uMag), small);

  const DblType dUedpFac =
    -1.0 / (Ue * density) * stk::math::pow(pressureRatio, -1.0 / kappa);
  DblType dUeds = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    dUeds += velocity_.get(node, i) / uMag * dUedpFac * dpdx_.get(node, i);
  }

  // Fixed-point iteration for the pressure gradient parameter lambda_t;
  // AHD correlation is valid for -0.068253 <= lambda_t <= 0.1
  const DblType lambda_t_min = -0.068253;
  const DblType lambda_t_max = 0.1;

  DblType lambda_t = 0.0;
  DblType theta_t = 0.0;
  DblType h12 = 0.0;
  DblType pi_shape = 0.0;

  const int iter_max = 15;
  for (int iter = 1; iter <= iter_max; ++iter) {
    const DblType theta_t_old = theta_t;

    if (lambda_t >= 0.0) {
      h12 = 4.02923 - stk::math::sqrt(
                        -8838.4 * stk::math::pow(lambda_t, 4) +
                        1105.1 * stk::math::pow(lambda_t, 3) -
                        67.962 * stk::math::pow(lambda_t, 2) +
                        17.574 * lambda_t + 2.0593);
    } else {
      h12 = 2.072 + 0.0731 / (lambda_t + 0.14);
    }

    pi_shape = 0.071665 * stk::math::pow(h12, 3) -
               0.73186 * stk::math::pow(h12, 2) + 4.2563 * h12 - 5.1743;

    const DblType ReTheta = Rev / stk::math::max(pi_shape, small);
    theta_t = ReTheta * nue / Ue;
    lambda_t = ReTheta * ReTheta * nue / (Ue * Ue) * dUeds;
    lambda_t =
      stk::math::max(stk::math::min(lambda_t, lambda_t_max), lambda_t_min);

    if (iter > 10 && stk::math::abs(theta_t - theta_t_old) <= 1.0e-10) {
      break;
    }
  }

  const DblType Ret = -(177.0 * Me * Me - 22.0 * Me + 210.0) *
                      stk::math::log((7.0 * Me + 4.8) * TuL / 100.0) *
                      stk::math::exp((5.0 * Me + 27.0) * lambda_t);

  const DblType fonset1 =
    Rev / stk::math::max(pi_shape, small) / stk::math::max(Ret, small);
  const DblType fonset2 =
    stk::math::min(stk::math::max(fonset1, stk::math::pow(fonset1, 4)), 2.0);
  const DblType rt = tvisc / visc;
  const DblType fonset3 =
    stk::math::max(1.0 - stk::math::pow(rt / 2.0, 3), 0.0);
  const DblType fonset = stk::math::max(fonset2 - fonset3, 0.0);
  const DblType fturb = stk::math::exp(-stk::math::pow(rt / 4.0, 4));

  const DblType Pgamma = flength * density * sijMag * fonset * (1.0 - gamint);
  const DblType Dgamma =
    caTwo * density * vortMag * fturb * gamint * (ceTwo * gamint - 1.0);

  // Jacobians dP/dgamma and dD/dgamma
  const DblType PgammaDir = -flength * density * sijMag * fonset;
  const DblType DgammaDir =
    caTwo * density * vortMag * fturb * (2.0 * ceTwo * gamint - 1.0);

  rhs(0) += (Pgamma - Dgamma) * dVol;
  lhs(0, 0) += stk::math::max(DgammaDir - PgammaDir, 0.0) * dVol;
}

} // namespace kynema_ugf
} // namespace sierra
