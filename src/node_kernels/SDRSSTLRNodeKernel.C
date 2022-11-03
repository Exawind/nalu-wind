// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SDRSSTLRNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

SDRSSTLRNodeKernel::SDRSSTLRNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<SDRSSTLRNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    nDim_(meta.spatial_dimension())
{
}

void
SDRSSTLRNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
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
  sstLRDestruct_ = realm.get_turb_model_constant(TM_sstLRDestruct);
  sstLRProd_ = realm.get_turb_model_constant(TM_sstLRProd);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
SDRSSTLRNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = stk::math::max(tke_.get(node, 0), 1.0e-12);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
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

  DblType chi_numer = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      for (int k = 0; k < nDim_; ++k) {
        const auto rot_ij = 0.5 * (dudx_.get(node, i * nDim_ + j) -
                                   dudx_.get(node, j * nDim_ + i));
        const auto rot_jk = 0.5 * (dudx_.get(node, j * nDim_ + k) -
                                   dudx_.get(node, k * nDim_ + j));
        const auto str_ki = 0.5 * (dudx_.get(node, k * nDim_ + i) +
                                   dudx_.get(node, i * nDim_ + k));
        chi_numer += rot_ij * rot_jk * str_ki;
      }
    }
  }

  const DblType chi_omega = stk::math::abs(
    chi_numer / stk::math::pow(0.09 * stk::math::max(sdr, 1.e-8), 3.0));
  const DblType beta =
    0.072 * (1.0 + 70.0 * chi_omega) / (1.0 + 80.0 * chi_omega);

  // JAM: Changes for SWH LowRe
  const DblType alpha0_star = 0.072 / 3.0;
  const DblType alpha_inf = 0.52;
  const DblType alpha0 = 1.0 / 9.0;
  const DblType Rk = 6.0;
  const DblType Rw = 2.95;
  const DblType ReT = density * tke / sdr / visc;
  const DblType Rbeta = 8.0;
  const DblType betaStarLowRe =
    betaStar_ * (4.0 / 15.0 + stk::math::pow(ReT / Rbeta, 4.0)) /
    (1.0 + stk::math::pow(ReT / Rbeta, 4.0));

  // JAM: Added for SWH LowRe
  const DblType alpha_star = (alpha0_star + ReT / Rk) / (1.0 + ReT / Rk);
  const DblType alpha =
    (alpha_inf / alpha_star) * ((alpha0 + ReT / Rw) / (1.0 + ReT / Rw));

  // Blend constants for SDR
  const DblType omf1 = (1.0 - fOneBlend);
  const DblType betaBlend =
    sstLRDestruct_ * (fOneBlend * beta + omf1 * betaTwo_) +
    (1.0 - sstLRDestruct_) * (fOneBlend * betaOne_ + omf1 * betaTwo_);
  const DblType gamma =
    sstLRProd_ * (fOneBlend * alpha + omf1 * gammaTwo_) +
    (1.0 - sstLRProd_) * (fOneBlend * gammaOne_ + omf1 * gammaTwo_);
  const DblType sigmaD = 2.0 * omf1 * sigmaWTwo_;
  const DblType betaStarBlend = fOneBlend * betaStarLowRe + omf1 * betaStar_;

  const DblType Dk = betaStarBlend * density * sdr * tke;

  // Clip production term and clip negative productions
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, stk::math::max(Pk, 0.0));

  // Pw includes 1/tvisc scaling; tvisc may be zero at a dirichlet low Re
  // approach (clip)
  // JAM: Changes for SWH LowRe, check densities...
  const DblType Pw = gamma * Pk * sdr / stk::math::max(tke, 1.e-12);
  const DblType Dw = betaBlend * density * sdr * sdr;
  const DblType Sw = sigmaD * density * crossDiff / sdr;

  // SUST source term
  const DblType Dwamb = betaBlend * density * sdrAmb_ * sdrAmb_;

  rhs(0) += (Pw - Dw + Dwamb + Sw) * dVol;
  lhs(0, 0) +=
    (2.0 * betaBlend * density * sdr + stk::math::max(Sw / sdr, 0.0)) * dVol;
}

} // namespace nalu
} // namespace sierra
