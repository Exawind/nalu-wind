// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SDRSSTDESNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

SDRSSTDESNodeKernel::SDRSSTDESNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<SDRSSTDESNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    cellLengthScaleID_(get_field_ordinal(meta, "sst_max_length_scale")),
    nDim_(meta.spatial_dimension())
{
}

void
SDRSSTDESNodeKernel::setup(Realm& realm)
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
  cellLengthScale_ = fieldMgr.get_field<double>(cellLengthScaleID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  sigmaWTwo_ = realm.get_turb_model_constant(TM_sigmaWTwo);
  betaOne_ = realm.get_turb_model_constant(TM_betaOne);
  betaTwo_ = realm.get_turb_model_constant(TM_betaTwo);
  gammaOne_ = realm.get_turb_model_constant(TM_gammaOne);
  gammaTwo_ = realm.get_turb_model_constant(TM_gammaTwo);
  cDESke_ = realm.get_turb_model_constant(TM_cDESke);
  cDESkw_ = realm.get_turb_model_constant(TM_cDESkw);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
SDRSSTDESNodeKernel::execute(
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

  // Blend constants for SDR
  const DblType omf1 = (1.0 - fOneBlend);
  const DblType beta = fOneBlend * betaOne_ + omf1 * betaTwo_;
  const DblType gamma = fOneBlend * gammaOne_ + omf1 * gammaTwo_;
  const DblType sigmaD = 2.0 * omf1 * sigmaWTwo_;
  const DblType cDES = omf1 * cDESke_ + fOneBlend * cDESkw_;

  const DblType small = 1.0e-16;
  const DblType eddyLengthRANS =
    stk::math::sqrt(tke) / stk::math::max(betaStar_ * sdr, small);
  const DblType eddyLengthLES = cDES * cellLengthScale_.get(node, 0);
  const DblType eddyLengthDES = stk::math::min(eddyLengthRANS, eddyLengthLES);

  const DblType Dk =
    density * stk::math::sqrt(tke) * tke / stk::math::max(eddyLengthDES, small);

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);

  // Production term with appropriate clipping of tvisc
  const DblType Pw = gamma * density * Pk / stk::math::max(tvisc, small);
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
