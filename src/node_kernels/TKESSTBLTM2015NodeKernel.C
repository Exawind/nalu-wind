// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKESSTBLTM2015NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKESSTBLTM2015NodeKernel::TKESSTBLTM2015NodeKernel(
  const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKESSTBLTM2015NodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    gamintID_(get_field_ordinal(meta, "gamma_transition")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTBLTM2015NodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  gamint_ = fieldMgr.get_field<double>(gamintID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  wallDist_ = fieldMgr.get_field<double>(wallDistID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  tkeAmb_ = realm.get_turb_model_constant(TM_tkeAmb);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
TKESSTBLTM2015NodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  const DblType gamint = gamint_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType dw = wallDist_.get(node, 0);

  const DblType Ck_BLT = 1.0;
  const DblType CSEP = 1.0;
  const DblType Retclim = 1100.0;

  DblType Pklim = 0.0;
  DblType sijMag = 1.0e-16;
  DblType vortMag = 1.0e-16;

  DblType Pk = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    // const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      // const auto dudxij = dudx_.get(node, offset + j);
      const double duidxj = dudx_.get(node, nDim_ * i + j);
      const double dujdxi = dudx_.get(node, nDim_ * j + i);

      const double rateOfStrain = 0.5 * (duidxj + dujdxi);
      const double vortTensor = 0.5 * (duidxj - dujdxi);

      sijMag += rateOfStrain * rateOfStrain;
      vortMag += vortTensor * vortTensor;
    }
  }

  sijMag = stk::math::sqrt(2.0 * sijMag);
  vortMag = stk::math::sqrt(2.0 * vortMag);

  DblType Rev = density * dw * dw * sijMag / visc;
  DblType Fonlim =
    stk::math::min(stk::math::max(Rev / 2.2 / Retclim - 1.0, 0.0), 3.0);

  // Pk based on Kato-Launder formulation
  Pk = gamint * tvisc * sijMag * vortMag;
  Pklim = 5.0 * Ck_BLT * stk::math::max(gamint - 0.2, 0.0) * (1.0 - gamint) *
          Fonlim * stk::math::max(3.0 * CSEP * visc - tvisc, 0.0) * sijMag *
          vortMag;
  const DblType Dk =
    betaStar_ * density * sdr * tke * stk::math::max(gamint, 0.1);

  // Clip production term and prevent Pk from being negative:
  // Deactivated w/  Kato-Launder formulation
  // Pk = stk::math::min(tkeProdLimitRatio_ * Dk, stk::math::max(Pk, 0.0));

  // SUST source term
  const DblType Dkamb = betaStar_ * density * sdrAmb_ * tkeAmb_;

  rhs(0) += (Pk + Pklim - Dk + Dkamb) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * stk::math::max(gamint, 0.1) * dVol;
}

} // namespace nalu
} // namespace sierra
