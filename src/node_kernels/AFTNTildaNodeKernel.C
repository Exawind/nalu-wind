// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "node_kernels/AFTNTildaNodeKernel.h"
#include "Realm.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace kynema_ugf {

AFTNTildaNodeKernel::AFTNTildaNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<AFTNTildaNodeKernel>(),
    densityID_(get_field_ordinal(meta, "density")),
    viscosityID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    minDistanceID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    dwalldistdxID_(get_field_ordinal(meta, "dwalldistdx")),
    dnDotVdxID_(get_field_ordinal(meta, "dnDotVdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
AFTNTildaNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  density_ = fieldMgr.get_field<double>(densityID_);
  viscosity_ = fieldMgr.get_field<double>(viscosityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  minDistance_ = fieldMgr.get_field<double>(minDistanceID_);
  dwalldistdx_ = fieldMgr.get_field<double>(dwalldistdxID_);
  dnDotVdx_ = fieldMgr.get_field<double>(dnDotVdxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
}

KOKKOS_FUNCTION
void
AFTNTildaNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType rho = density_.get(node, 0);
  const DblType mu = viscosity_.get(node, 0);
  const DblType muT = tvisc_.get(node, 0);
  const DblType wallDistance = minDistance_.get(node, 0);
  const DblType dualVolume = dualNodalVolume_.get(node, 0);
  const DblType nu = mu / stk::math::max(rho, 1.0e-16);

  DblType dvnn = 0.0;
  DblType strainSquared = 0.0;
  DblType vorticitySquared = 0.0;
  for (int i = 0; i < nDim_; ++i)
    dvnn += dwalldistdx_.get(node, i) * dnDotVdx_.get(node, i);

  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      const DblType duidxj = dudx_.get(node, nDim_ * i + j);
      const DblType dujdxi = dudx_.get(node, nDim_ * j + i);
      const DblType strain = 0.5 * (duidxj + dujdxi);
      const DblType vorticity = 0.5 * (duidxj - dujdxi);
      strainSquared += strain * strain;
      vorticitySquared += vorticity * vorticity;
    }
  }

  const DblType strainMagnitude = stk::math::sqrt(2.0 * strainSquared);
  const DblType vorticityMagnitude = stk::math::sqrt(2.0 * vorticitySquared);

  const DblType hLocal =
    wallDistance * wallDistance * dvnn / stk::math::max(nu, 1.0e-16);
  const DblType h12 =
    stk::math::min(stk::math::max(0.26 * hLocal + 2.4, 2.2), 20.0);
  const DblType h12MinusOne = stk::math::max(h12 - 1.0, 1.0e-16);

  const DblType reThetaZero = stk::math::pow(
    10.0, 0.7 * stk::math::tanh(14.0 / h12MinusOne - 9.24) +
            2.492 / stk::math::pow(h12MinusOne, 0.43) + 0.62);
  const DblType kV = 1.0 / (0.4036 * h12 * h12 - 2.5394 * h12 + 4.3273);
  const DblType reVZero = kV * reThetaZero;
  const DblType reV = rho * strainMagnitude * wallDistance * wallDistance /
                      stk::math::max(mu + muT, 1.0e-16);
  const DblType fCrit = (reV >= reVZero) ? 1.0 : 0.0;

  const DblType lH12 =
    stk::math::max((6.54 * h12 - 14.07) / (h12 * h12), 1.0e-16);
  const DblType DH12 = 2.4 * h12 / h12MinusOne;
  const DblType mH12 =
    (0.058 * (h12 - 4.0) * (h12 - 4.0) / h12MinusOne - 0.068) / lH12;
  const DblType fgrowth = DH12 * (1.0 + mH12) * 0.5 * lH12;
  const DblType dnDReTheta =
    0.028 * h12MinusOne -
    0.0345 * stk::math::exp(-stk::math::pow(3.87 / h12MinusOne - 2.52, 2.0));

  const DblType production =
    rho * vorticityMagnitude * fCrit * fgrowth * dnDReTheta;
  rhs(0) += production * dualVolume;
}

} // namespace kynema_ugf
} // namespace sierra
