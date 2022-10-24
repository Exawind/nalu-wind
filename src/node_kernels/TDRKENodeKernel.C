// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TDRKENodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TDRKENodeKernel::TDRKENodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TDRKENodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    tdrID_(get_field_ordinal(meta, "total_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dplusID_(get_field_ordinal(meta, "dplus_wall_function")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
TDRKENodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  tdr_ = fieldMgr.get_field<double>(tdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  wallDist_ = fieldMgr.get_field<double>(wallDistID_);
  dplus_ = fieldMgr.get_field<double>(dplusID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  // Update turbulence model constants
  cEpsOne_ = realm.get_turb_model_constant(TM_cEpsOne);
  cEpsTwo_ = realm.get_turb_model_constant(TM_cEpsTwo);
  fOne_ = realm.get_turb_model_constant(TM_fOne);
}

KOKKOS_FUNCTION
void
TDRKENodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType tdr = tdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType wallDist = wallDist_.get(node, 0);
  const DblType dplus = dplus_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  DblType Pk = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset + j);
      Pk += dudxij * (dudxij + dudx_.get(node, j * nDim_ + i));
    }
  }
  Pk *= tvisc;

  const double Re_t = density * tke * tke / visc / stk::math::max(tdr, 1.0e-16);
  const double fTwo = 1.0 - 0.4 / 1.8 * stk::math::exp(-Re_t * Re_t / 36.0);

  // Production term with appropriate clipping of tvisc
  const DblType Pe = cEpsOne_ * fOne_ * Pk * tdr / stk::math::max(tke, 1.0e-16);
  const DblType DeFac =
    cEpsTwo_ * fTwo * density * tdr / stk::math::max(tke, 1.0e-16);
  const DblType De = DeFac * tdr;
  const DblType LeFac = 2.0 * visc * stk::math::exp(-0.5 * dplus) /
                        stk::math::max(wallDist * wallDist, 1.0e-16);
  const DblType Le = -LeFac * tdr;

  rhs(0) += (Pe - De + Le) * dVol;
  lhs(0, 0) += (2.0 * DeFac + LeFac) * dVol;
}

} // namespace nalu
} // namespace sierra
