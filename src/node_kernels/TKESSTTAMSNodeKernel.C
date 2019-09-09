/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/TKESSTTAMSNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

TKESSTTAMSNodeKernel::TKESSTTAMSNodeKernel(
  const stk::mesh::MetaData& meta, const std::string coordsName)
  : NGPNodeKernel<TKESSTTAMSNodeKernel>(),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    coordinatesID_(get_field_ordinal(meta, coordsName)),
    viscID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    tkeNp1ID_(get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1)),
    sdrNp1ID_(get_field_ordinal(
      meta, "specific_dissipation_rate", stk::mesh::StateNP1)),
    prodID_(get_field_ordinal(meta, "average_production")),
    densityID_(get_field_ordinal(meta, "average_density")),
    nDim_(meta.spatial_dimension())
{
}

void
TKESSTTAMSNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  viscosity_ = fieldMgr.get_field<double>(viscID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  rho_ = fieldMgr.get_field<double>(densityID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  prod_ = fieldMgr.get_field<double>(prodID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
}

void
TKESSTTAMSNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  NodeKernelTraits::DblType Pk = prod_.get(node, 0);

  const NodeKernelTraits::DblType tkeFac =
    betaStar_ * rho_.get(node, 0) * sdr_.get(node, 0);
  NodeKernelTraits::DblType Dk = tkeFac * stk::math::max(tke_.get(node, 0), 1.0e-12);

  Pk = stk::math::min(Pk, tkeProdLimitRatio_ * Dk);

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  rhs(0) += (Pk - Dk) * dualVolume;

  lhs(0, 0) += tkeFac * dualVolume;
}

} // namespace nalu
} // namespace sierra
