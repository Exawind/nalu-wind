// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SDRSSTNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

SDRSSTNodeKernel::SDRSSTNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<SDRSSTNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    nDim_(meta.spatial_dimension())
{
}

void
SDRSSTNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dkdx_ = fieldMgr.get_field<double>(dkdxID_);
  dwdx_ = fieldMgr.get_field<double>(dwdxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  fOneBlend_ = fieldMgr.get_field<double>(fOneBlendID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  sigmaWTwo_ = realm.get_turb_model_constant(TM_sigmaWTwo);
  betaOne_ = realm.get_turb_model_constant(TM_betaOne);
  betaTwo_ = realm.get_turb_model_constant(TM_betaTwo);
  gammaOne_ = realm.get_turb_model_constant(TM_gammaOne);
  gammaTwo_ = realm.get_turb_model_constant(TM_gammaTwo);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);

  lengthScaleLimiter_ = realm.solutionOptions_->lengthScaleLimiter_;
  if (lengthScaleLimiter_) {
    const NodeKernelTraits::DblType earthAngularVelocity =
      realm.solutionOptions_->earthAngularVelocity_;
    const NodeKernelTraits::DblType pi = std::acos(-1.0);
    const NodeKernelTraits::DblType latitude =
      realm.solutionOptions_->latitude_ * pi / 180.0;
    corfac_ = 2.0 * earthAngularVelocity * std::sin(latitude);
    referenceVelocity_ = realm.solutionOptions_->referenceVelocity_;
  }
}

KOKKOS_FUNCTION
void
SDRSSTNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  const DblType fOneBlend = fOneBlend_.get(node, 0);

  DblType Pk = 0.0;
  DblType crossDiff = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    crossDiff += dkdx_.get(node, i) * dwdx_.get(node, i);
    const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset + j);
      Pk += dudxij * (dudxij + dudx_.get(node, j * nDim_ + i));
    }
  }
  Pk *= tvisc;

  const DblType Dk = betaStar_ * density * sdr * tke;

  // Clip production term and clip negative productions
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, stk::math::max(Pk, 0.0));

  // Blend constants for SDR
  const DblType omf1 = (1.0 - fOneBlend);
  const DblType beta = fOneBlend * betaOne_ + omf1 * betaTwo_;
  const DblType sigmaD = 2.0 * omf1 * sigmaWTwo_;

  DblType gammaOne_apply;
  DblType gammaTwo_apply;
  // apply limiter to gamma
  if (lengthScaleLimiter_) {
    // calculate mixing length
    const DblType l_t =
      stk::math::sqrt(tke) / (stk::math::pow(betaStar_, .25) * sdr);

    // calculate maximum mixing length
    // the proportionality constant (.00027) was found by fitting to
    // measurements of atmospheric conditions as described in ref. Kob13
    const DblType l_e = .00027 * referenceVelocity_ / corfac_;

    // apply limiter to cEpsOne -> calculate gammaOne
    const DblType cEpsOne_one = gammaOne_ + 1.;
    const DblType cEpsTwo_one = betaOne_ / betaStar_ + 1.;
    const DblType cEpsOneStar_one =
      cEpsOne_one + (cEpsTwo_one - cEpsOne_one) * (l_t / l_e);
    gammaOne_apply = cEpsOneStar_one - 1.;

    // apply limiter to cEpsTwo -> calculate gammaTwo
    const DblType cEpsOne_two = gammaTwo_ + 1.;
    const DblType cEpsTwo_two = betaTwo_ / betaStar_ + 1.;
    const DblType cEpsOneStar_two =
      cEpsOne_two + (cEpsTwo_two - cEpsOne_two) * (l_t / l_e);
    gammaTwo_apply = cEpsOneStar_two - 1.;
  } else {
    gammaOne_apply = gammaOne_;
    gammaTwo_apply = gammaTwo_;
  }
  const DblType gamma = fOneBlend * gammaOne_apply + omf1 * gammaTwo_apply;

  // Production term with appropriate clipping of tvisc
  const DblType Pw = gamma * density * Pk / stk::math::max(tvisc, 1.0e-16);
  const DblType Dw = beta * density * sdr * sdr;
  const DblType Sw = sigmaD * density * crossDiff / sdr;

  // SUST source term
  const DblType Dwamb = beta * density * sdrAmb_ * sdrAmb_;

  rhs(0) += (Pw - Dw + Dwamb + Sw) * dVol;
  lhs(0, 0) +=
    (2.0 * beta * density * sdr + stk::math::max(Sw / sdr, 0.0)) * dVol;
}

} // namespace nalu
} // namespace sierra
