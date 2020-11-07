// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SDRSSTAMSNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

SDRSSTAMSNodeKernel::SDRSSTAMSNodeKernel(
  const stk::mesh::MetaData& meta, const std::string coordsName)
  : NGPNodeKernel<SDRSSTAMSNodeKernel>(),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    coordinatesID_(get_field_ordinal(meta, coordsName)),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    tkeNp1ID_(get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1)),
    sdrNp1ID_(get_field_ordinal(
      meta, "specific_dissipation_rate", stk::mesh::StateNP1)),
    betaID_(get_field_ordinal(meta, "k_ratio")),
    fOneBlendID_(get_field_ordinal(meta, "sst_f_one_blending")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    prodID_(get_field_ordinal(meta, "average_production")),
    densityID_(get_field_ordinal(meta, "density")),
    nDim_(meta.spatial_dimension())
{
}

void
SDRSSTAMSNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  rho_ = fieldMgr.get_field<double>(densityID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  beta_ = fieldMgr.get_field<double>(betaID_);
  prod_ = fieldMgr.get_field<double>(prodID_);
  fOneBlend_ = fieldMgr.get_field<double>(fOneBlendID_);
  dkdx_ = fieldMgr.get_field<double>(dkdxID_);
  dwdx_ = fieldMgr.get_field<double>(dwdxID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  sigmaWTwo_ = realm.get_turb_model_constant(TM_sigmaWTwo);
  betaOne_ = realm.get_turb_model_constant(TM_betaOne);
  betaTwo_ = realm.get_turb_model_constant(TM_betaTwo);
  gammaOne_ = realm.get_turb_model_constant(TM_gammaOne);
  gammaTwo_ = realm.get_turb_model_constant(TM_gammaTwo);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
}

void
SDRSSTAMSNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const NodeKernelTraits::DblType rho = rho_.get(node, 0);
  const NodeKernelTraits::DblType sdr = sdr_.get(node, 0);
  const NodeKernelTraits::DblType tke =
    stk::math::max(tke_.get(node, 0), 1.0e-12);
  const NodeKernelTraits::DblType tvisc = tvisc_.get(node, 0);
  const NodeKernelTraits::DblType fOneBlend = fOneBlend_.get(node, 0);

  NodeKernelTraits::DblType crossDiff = 0.0;
  for (int d = 0; d < nDim_; ++d)
    crossDiff += dkdx_.get(node, d) * dwdx_.get(node, d);

  // Clip negative productions, consistent with TKE
  NodeKernelTraits::DblType Pk = stk::math::max(prod_.get(node, 0), 0.0);
  const NodeKernelTraits::DblType Dk = betaStar_ * rho * sdr * tke;
  Pk = stk::math::min(Pk, tkeProdLimitRatio_ * Dk);

  // start the blending and constants
  const NodeKernelTraits::DblType om_fOneBlend = 1.0 - fOneBlend;
  const NodeKernelTraits::DblType beta =
    fOneBlend * betaOne_ + om_fOneBlend * betaTwo_;
  const NodeKernelTraits::DblType gamma =
    fOneBlend * gammaOne_ + om_fOneBlend * gammaTwo_;
  const NodeKernelTraits::DblType sigmaD = 2.0 * om_fOneBlend * sigmaWTwo_;

  // Pw includes 1/tvisc scaling; tvisc may be zero at a dirichlet low Re
  // approach (clip)
  const NodeKernelTraits::DblType Pw =
    gamma * rho * Pk / stk::math::max(tvisc, 1.0e-16);
  const NodeKernelTraits::DblType Dw = beta * rho * sdr * sdr;
  const NodeKernelTraits::DblType Sw = sigmaD * rho * crossDiff / sdr;

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  rhs(0) += (Pw - Dw + Sw) * dualVolume;

  lhs(0, 0) +=
    (2.0 * beta * rho * sdr + stk::math::max(Sw / sdr, 0.0)) * dualVolume;
}

} // namespace nalu
} // namespace sierra
