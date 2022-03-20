// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SDRSSTBLTM2015NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"
#include "NaluEnv.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

SDRSSTBLTM2015NodeKernel::SDRSSTBLTM2015NodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<SDRSSTBLTM2015NodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    minDID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    coordinatesID_(get_field_ordinal(meta, "coordinates")),
    velocityNp1ID_(get_field_ordinal(meta, "velocity")),
    nDim_(meta.spatial_dimension())
{
}

void
SDRSSTBLTM2015NodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dkdx_ = fieldMgr.get_field<double>(dkdxID_);
  dwdx_ = fieldMgr.get_field<double>(dwdxID_);
  minD_ = fieldMgr.get_field<double>(minDID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  fOneBlend_ = fieldMgr.get_field<double>(fOneBlendID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  velocityNp1_ = fieldMgr.get_field<double>(velocityNp1ID_);

  const std::string dofName = "specific_dissipation_rate";
  relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);
  sdrFreestream = realm.sdrFS;

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  sigmaWTwo_ = realm.get_turb_model_constant(TM_sigmaWTwo);
  betaOne_ = realm.get_turb_model_constant(TM_betaOne);
  betaTwo_ = realm.get_turb_model_constant(TM_betaTwo);
  gammaOne_ = realm.get_turb_model_constant(TM_gammaOne);
  gammaTwo_ = realm.get_turb_model_constant(TM_gammaTwo);
  c0t_ = realm.get_turb_model_constant(TM_c0t);
  xcoordEndFixedTurb_ = realm.solutionOptions_->xcoordEndFixedTurb_;
}

void
SDRSSTBLTM2015NodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  NALU_ALIGNED NodeKernelTraits::DblType coords[NodeKernelTraits::NDimMax]; // coordinates
  NALU_ALIGNED NodeKernelTraits::DblType vel[NodeKernelTraits::NDimMax];

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType visc = tvisc_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType minD      = minD_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  const DblType fOneBlend = fOneBlend_.get(node, 0);

  DblType Pk = 0.0;
  DblType Dk = 0.0;
  DblType crossDiff = 0.0;
  DblType sdrForcing = 0.0;
  DblType tc = 0.0;
  DblType sijMag    = 0.0;
  DblType vortMag   = 0.0;
  DblType velMag2 = 0.0;

  for (int d = 0; d < nDim_; d++) {
    coords[d] = coordinates_.get(node, d);
    vel[d] = velocityNp1_.get(node, d);
  }

  for (int i=0; i < nDim_; ++i) {
    crossDiff += dkdx_.get(node, i) * dwdx_.get(node, i);
    const int offset = nDim_ * i;
    for (int j=0; j < nDim_; ++j) {
     const double duidxj = dudx_.get(node, nDim_ * i + j);
     const double dujdxi = dudx_.get(node, nDim_ * j + i);

     const double rateOfStrain = 0.5 * (duidxj + dujdxi);
     const double vortTensor = 0.5 * (duidxj - dujdxi);
     sijMag += rateOfStrain * rateOfStrain;
     vortMag += vortTensor * vortTensor;
    }
  }

  sijMag = stk::math::sqrt(2.0*sijMag);
  vortMag = stk::math::sqrt(2.0*vortMag);
  velMag2 = vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2] + 1.e-14;

  Dk = betaStar_ * density * sdr * tke;
  Pk = tvisc * sijMag * vortMag; // Pk based on Kato-Launder formulation. Recommended in Menter (2015) to avoid excessive levels of TKE in stagnation regions

  if (coords[0] < xcoordEndFixedTurb_) {
    tc = 500.0 * visc / density / velMag2;
    sdrForcing = c0t_ * density * (sdrFreestream - sdr) / tc;
    rhs(0) += sdrForcing * dVol;
    lhs(0, 0) += c0t_ * density * dVol/ tc;
  }
  else {
    // Blend constants for SDR
    const DblType ry = density * minD * stk::math::sqrt(tke)/visc;
    const DblType arg = ry / 120.0;
    const DblType f3 = stk::math::exp(-arg*arg*arg*arg*arg*arg*arg*arg);
    const DblType fOneBlendBLT = stk::math::max( fOneBlend, f3);

    const DblType omf1 = (1.0 - fOneBlend);
    const DblType beta = fOneBlend * betaOne_ + omf1 * betaTwo_;
    const DblType gamma = fOneBlend * gammaOne_ + omf1 * gammaTwo_;
    const DblType sigmaD = 2.0 * omf1 * sigmaWTwo_;

    // Production term with appropriate clipping of tvisc
    const DblType Pw = gamma * density * Pk / stk::math::max(tvisc, 1.0e-16);
    const DblType Dw = beta * density * sdr * sdr;
    const DblType Sw = sigmaD * density * crossDiff / sdr;

    rhs(0) += (Pw - Dw + Sw + sdrForcing) * dVol;
    lhs(0, 0) += (2.0 * beta * density * sdr + stk::math::max(Sw / sdr, 0.0)) * dVol;
  }
}

} // namespace nalu

} // namespace sierra
