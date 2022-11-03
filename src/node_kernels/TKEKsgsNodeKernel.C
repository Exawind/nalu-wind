// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKEKsgsNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "SimdInterface.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKEKsgsNodeKernel::TKEKsgsNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<TKEKsgsNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
TKEKsgsNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  // Update turbulence model constants
  cEps_ = realm.get_turb_model_constant(TM_cEps);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
}

KOKKOS_FUNCTION
void
TKEKsgsNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  const DblType filter = std::pow(dVol, 1.0 / nDim_);

  DblType Pk = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const int offset = nDim_ * i;
    for (int j = 0; j < nDim_; ++j) {
      const auto dudxij = dudx_.get(node, offset + j);
      Pk += dudxij * (dudxij + dudx_.get(node, j * nDim_ + i));
    }
  }
  Pk *= tvisc;

  const DblType Dk = cEps_ * density * stk::math::pow(tke, 1.5) / filter;

  // Clip production term
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, Pk);

  rhs(0) += (Pk - Dk) * dVol;
  lhs(0, 0) += 1.5 * cEps_ * density * stk::math::sqrt(tke) / filter * dVol;
}

} // namespace nalu
} // namespace sierra
