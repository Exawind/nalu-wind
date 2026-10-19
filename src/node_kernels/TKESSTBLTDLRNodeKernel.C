// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKESSTBLTDLRNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace kynema_ugf {

TKESSTBLTDLRNodeKernel::TKESSTBLTDLRNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKESSTBLTDLRNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTBLTDLRNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  gamint_ = fieldMgr.get_field<double>(gamintID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  tkeAmb_ = realm.get_turb_model_constant(TM_tkeAmb);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
TKESSTBLTDLRNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType gamint = gamint_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  DblType sijMag = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      const DblType rateOfStrain =
        0.5 * (dudx_.get(node, nDim_ * i + j) + dudx_.get(node, nDim_ * j + i));
      sijMag += rateOfStrain * rateOfStrain;
    }
  }
  sijMag = stk::math::sqrt(2.0 * sijMag);

  // SST-2003 production (strain-based, limited) scaled by intermittency
  const DblType Pk =
    gamint * stk::math::min(
               tvisc * sijMag * sijMag,
               tkeProdLimitRatio_ * betaStar_ * density * sdr * tke);

  const DblType gamDest = stk::math::min(stk::math::max(gamint, 0.1), 1.0);
  const DblType Dk = betaStar_ * density * sdr * tke * gamDest;

  // SUST source term
  const DblType Dkamb = betaStar_ * density * sdrAmb_ * tkeAmb_;

  rhs(0) += (Pk - Dk + Dkamb) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * gamDest * dVol;
}

} // namespace kynema_ugf
} // namespace sierra
