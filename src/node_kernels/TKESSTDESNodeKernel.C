// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKESSTDESNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "SimdInterface.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKESSTDESNodeKernel::TKESSTDESNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKESSTDESNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    maxLenScaleID_(get_field_ordinal(meta, "sst_max_length_scale")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTDESNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  maxLenScale_ = fieldMgr.get_field<double>(maxLenScaleID_);
  fOneBlend_ = fieldMgr.get_field<double>(fOneBlendID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  cDESke_ = realm.get_turb_model_constant(TM_cDESke);
  cDESkw_ = realm.get_turb_model_constant(TM_cDESkw);
  tkeAmb_ = realm.get_turb_model_constant(TM_tkeAmb);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
TKESSTDESNodeKernel::execute(
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
  const DblType maxLenScale = maxLenScale_.get(node, 0);
  const DblType fOneBlend = fOneBlend_.get(node, 0);

  DblType Pk = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset + j);
      Pk += dudxij * (dudxij + dudx_.get(node, j * nDim_ + i));
    }
  }
  Pk *= tvisc;

  // blend cDES constant
  const DblType cDES = fOneBlend * cDESkw_ + (1.0 - fOneBlend) * cDESke_;

  const DblType sqrtTke = stk::math::sqrt(tke);
  const DblType lSST = sqrtTke / betaStar_ / sdr;

  // Find minimum length scale, limit minimum value to 1.0e-16 to prevent
  // division by zero later on
  const DblType lDES =
    stk::math::max(1.0e-16, stk::math::min(lSST, cDES * maxLenScale));

  DblType Dk = density * tke * sqrtTke / lDES;

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);

  // SUST source term
  const DblType sqrtTkeAmb = stk::math::sqrt(tkeAmb_);
  const DblType lSSTAmb =
    sqrtTkeAmb / betaStar_ / stk::math::max(1.0e-16, sdrAmb_);
  const DblType lDESAmb = stk::math::max(
    1.0e-16, (lSST < cDES * maxLenScale) ? lSSTAmb : cDES * maxLenScale);
  const DblType Dkamb = density * tkeAmb_ * sqrtTkeAmb / lDESAmb;

  rhs(0) += (Pk - Dk + Dkamb) * dVol;
  lhs(0, 0) += 1.5 * density / lDES * sqrtTke * dVol;
}

} // namespace nalu
} // namespace sierra
