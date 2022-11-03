// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKEKENodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKEKENodeKernel::TKEKENodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKEKENodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    tdrID_(get_field_ordinal(meta, "total_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
TKEKENodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  tdr_ = fieldMgr.get_field<double>(tdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  wallDist_ = fieldMgr.get_field<double>(wallDistID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
}

KOKKOS_FUNCTION
void
TKEKENodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType tdr = tdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType wallDist = wallDist_.get(node, 0);
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

  DblType Dk = density * tdr;

  const DblType lFac =
    2.0 * visc / stk::math::max(wallDist * wallDist, 1.0e-16);
  DblType Lk = -lFac * tke;

  rhs(0) += (Pk - Dk + Lk) * dVol;
  lhs(0, 0) += lFac * dVol;
}

} // namespace nalu
} // namespace sierra
