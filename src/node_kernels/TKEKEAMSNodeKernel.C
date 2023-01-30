// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKEKEAMSNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKEKEAMSNodeKernel::TKEKEAMSNodeKernel(
  const stk::mesh::MetaData& meta, const std::string coordsName)
  : NGPNodeKernel<TKEKEAMSNodeKernel>(),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    tdrID_(get_field_ordinal(meta, "total_dissipation_rate")),
    prodID_(get_field_ordinal(meta, "average_production")),
    densityID_(get_field_ordinal(meta, "density")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    nDim_(meta.spatial_dimension())
{
}

void
TKEKEAMSNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  tdr_ = fieldMgr.get_field<double>(tdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  prod_ = fieldMgr.get_field<double>(prodID_);
  wallDist_ = fieldMgr.get_field<double>(wallDistID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

KOKKOS_FUNCTION
void
TKEKEAMSNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType tdr = tdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType prod = prod_.get(node, 0);
  const DblType wallDist = wallDist_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  DblType Dk = density * tdr;

  const DblType lFac =
    2.0 * visc / stk::math::max(wallDist * wallDist, 1.0e-16);
  DblType Lk = -lFac * tke;

  rhs(0) += (prod - Dk + Lk) * dVol;
  lhs(0, 0) += lFac * dVol;
}

} // namespace nalu
} // namespace sierra
