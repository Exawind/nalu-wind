// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TDRKEAMSNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TDRKEAMSNodeKernel::TDRKEAMSNodeKernel(
  const stk::mesh::MetaData& meta, const std::string /*coordsName*/)
  : NGPNodeKernel<TDRKEAMSNodeKernel>(),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    tdrID_(get_field_ordinal(meta, "total_dissipation_rate")),
    dplusID_(get_field_ordinal(meta, "dplus_wall_function")),
    wallDistID_(get_field_ordinal(meta, "minimum_distance_to_wall")),
    prodID_(get_field_ordinal(meta, "average_production")),
    densityID_(get_field_ordinal(meta, "density")),
    nDim_(meta.spatial_dimension())
{
}

void
TDRKEAMSNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  tdr_ = fieldMgr.get_field<double>(tdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  wallDist_ = fieldMgr.get_field<double>(wallDistID_);
  dplus_ = fieldMgr.get_field<double>(dplusID_);
  prod_ = fieldMgr.get_field<double>(prodID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  const std::string dofName = "total_dissipation_rate";
  // relaxFac_ = realm.solutionOptions_->get_relaxation_factor(dofName);

  // Update turbulence model constants
  cEpsOne_ = realm.get_turb_model_constant(TM_cEpsOne);
  cEpsTwo_ = realm.get_turb_model_constant(TM_cEpsTwo);
  fOne_ = realm.get_turb_model_constant(TM_fOne);
}

void
TDRKEAMSNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType tdr = tdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType wallDist = wallDist_.get(node, 0);
  const DblType dplus = dplus_.get(node, 0);
  const DblType prod = prod_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);

  const DblType Re_t =
    density * tke * tke / visc / stk::math::max(tdr, 1.0e-16);
  const DblType fTwo = 1.0 - 0.4 / 1.8 * stk::math::exp(-Re_t * Re_t / 36.0);

  // Production term with appropriate clipping of tvisc
  const DblType Pe =
    cEpsOne_ * fOne_ * prod * tdr / stk::math::max(tke, 1.0e-16);
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
