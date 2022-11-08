// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
#include "node_kernels/TKESSTIDDESNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "SimdInterface.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

TKESSTIDDESNodeKernel::TKESSTIDDESNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKESSTIDDESNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    maxLenScaleID_(get_field_ordinal(meta, "sst_max_length_scale")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    ransIndicatorID_(get_field_ordinal(meta, "iddes_rans_indicator")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTIDDESNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  wallDist_ = fieldMgr.get_field<double>(wallDistID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  maxLenScale_ = fieldMgr.get_field<double>(maxLenScaleID_);
  fOneBlend_ = fieldMgr.get_field<double>(fOneBlendID_);
  ransIndicator_ = fieldMgr.get_field<double>(ransIndicatorID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  // this is the cdes2 from the Gritskavich 2012 paper
  cDESke_ = realm.get_turb_model_constant(TM_cDESke);
  // this is the cdes1 from the Gritskavich 2012 paper
  cDESkw_ = realm.get_turb_model_constant(TM_cDESkw);
  kappa_ = realm.get_turb_model_constant(TM_kappa);
  iddes_Cw_ = realm.get_turb_model_constant(TM_iddes_Cw);
  iddes_Cdt1_ = realm.get_turb_model_constant(TM_iddes_Cdt1);
  iddes_Cdt2_ = realm.get_turb_model_constant(TM_iddes_Cdt2);
  iddes_Cl_ = realm.get_turb_model_constant(TM_iddes_Cl);
  iddes_Ct_ = realm.get_turb_model_constant(TM_iddes_Ct);
  tkeAmb_ = realm.get_turb_model_constant(TM_tkeAmb);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
TKESSTIDDESNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  const DblType dw = wallDist_.get(node, 0);
  const DblType maxLenScale = maxLenScale_.get(node, 0);
  const DblType fOneBlend = fOneBlend_.get(node, 0);

  DblType Pk = 0.0;
  DblType sijSq = 1.0e-16;
  DblType omegaSq = 1.0e-16;
  for (int i = 0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset + j);
      const DblType rateOfStrain =
        0.5 * (dudxij + dudx_.get(node, j * nDim_ + i));
      sijSq += rateOfStrain * rateOfStrain;
      const DblType rateOfOmega =
        0.5 * (dudxij - dudx_.get(node, j * nDim_ + i));
      omegaSq += rateOfOmega * rateOfOmega;
    }
  }
  sijSq *= 2.0;
  omegaSq *= 2.0;
  Pk = tvisc * sijSq;

  DblType denom =
    density * kappa_ * kappa_ * dw * dw *
    stk::math::max(stk::math::sqrt(0.5 * (sijSq + omegaSq)), 1e-10);
  DblType rdl = visc / denom;
  DblType rdt = tvisc / denom;
  DblType fl = stk::math::tanh(stk::math::pow(iddes_Cl_ * iddes_Cl_ * rdl, 10));
  DblType ft = stk::math::tanh(stk::math::pow(iddes_Ct_ * iddes_Ct_ * rdt, 3));
  DblType alpha = 0.25 - dw / maxLenScale;
  DblType fe1 = (alpha < 0) ? 2.0 * stk::math::exp(-9.0 * alpha * alpha)
                            : 2.0 * stk::math::exp(-11.09 * alpha * alpha);
  DblType fe2 = 1.0 - stk::math::max(ft, fl);
  DblType fe = fe2 * stk::math::max((fe1 - 1.0), 0.0);
  DblType fb = stk::math::min(2.0 * stk::math::exp(-9.0 * alpha * alpha), 1.0);
  DblType fdt =
    1.0 - stk::math::tanh(stk::math::pow(iddes_Cdt1_ * rdt, iddes_Cdt2_));
  DblType fdHat = stk::math::max((1.0 - fdt), fb);
  DblType delta =
    stk::math::min(iddes_Cw_ * stk::math::max(dw, maxLenScale), maxLenScale);

  // blend cDES constant
  const DblType cDES = fOneBlend * cDESkw_ + (1.0 - fOneBlend) * cDESke_;

  const DblType sqrtTke = stk::math::sqrt(tke);
  const DblType lSST = sqrtTke / betaStar_ / sdr;
  const DblType lLES = cDES * delta;

  // Find minimum length scale, limit minimum value to 1.0e-16 to prevent
  // division by zero later on
  const DblType ransInd = fdHat * (1.0 + fe);
  ransIndicator_.get(node, 0) = ransInd;

  const DblType lIDDES =
    stk::math::max(1.0e-16, ransInd * lSST + (1.0 - fdHat) * lLES);

  DblType Dk = density * tke * sqrtTke / lIDDES;

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);

  // SUST source term
  const DblType sqrtTkeAmb = stk::math::sqrt(tkeAmb_);
  const DblType lSSTAmb =
    sqrtTkeAmb / betaStar_ / stk::math::max(1.0e-16, sdrAmb_);
  const DblType lIDDESAmb =
    stk::math::max(1.0e-16, ransInd * lSSTAmb + (1.0 - fdHat) * lLES);
  const DblType Dkamb = density * tkeAmb_ * sqrtTkeAmb / lIDDESAmb;

  rhs(0) += (Pk - Dk + Dkamb) * dVol;
  lhs(0, 0) += 1.5 * density / lIDDES * sqrtTke * dVol;
}

} // namespace nalu
} // namespace sierra
