// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/SDRKOAMSNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

SDRKOAMSNodeKernel::SDRKOAMSNodeKernel(const stk::mesh::MetaData& meta)
  : NGPNodeKernel<SDRKOAMSNodeKernel>(),
    tkeID_(get_field_ordinal(meta, "turbulent_ke")),
    sdrID_(get_field_ordinal(meta, "specific_dissipation_rate")),
    densityID_(get_field_ordinal(meta, "density")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    viscID_(get_field_ordinal(meta, "viscosity")),
    dudxID_(get_field_ordinal(meta, "average_dudx")),
    dkdxID_(get_field_ordinal(meta, "dkdx")),
    dwdxID_(get_field_ordinal(meta, "dwdx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    prodID_(get_field_ordinal(meta, "average_production")),
    nDim_(meta.spatial_dimension())
{
}

void
SDRKOAMSNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  tke_ = fieldMgr.get_field<double>(tkeID_);
  sdr_ = fieldMgr.get_field<double>(sdrID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  visc_ = fieldMgr.get_field<double>(viscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dkdx_ = fieldMgr.get_field<double>(dkdxID_);
  dwdx_ = fieldMgr.get_field<double>(dwdxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  prod_ = fieldMgr.get_field<double>(prodID_);

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  sigmaWTwo_ = realm.get_turb_model_constant(TM_sigmaWTwo);
  betaOne_ = realm.get_turb_model_constant(TM_betaOne);
  betaTwo_ = realm.get_turb_model_constant(TM_betaTwo);
  gammaOne_ = realm.get_turb_model_constant(TM_gammaOne);
  gammaTwo_ = realm.get_turb_model_constant(TM_gammaTwo);
}

void
SDRKOAMSNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
  const DblType visc = visc_.get(node, 0);
  const DblType dVol = dualNodalVolume_.get(node, 0);
  DblType Pk = prod_.get(node, 0);

  DblType crossDiff = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    crossDiff += dkdx_.get(node, i) * dwdx_.get(node, i);
  }

  // FIXME: Is this going to work using average_dudx directly?
  DblType chi_numer = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    for (int j = 0; j < nDim_; ++j) {
      for (int k = 0; k < nDim_; ++k) {
        const auto rot_ij = 0.5 * (dudx_.get(node, i * nDim_ + j) -
                                   dudx_.get(node, j * nDim_ + i));
        const auto rot_jk = 0.5 * (dudx_.get(node, j * nDim_ + k) -
                                   dudx_.get(node, k * nDim_ + j));
        const auto str_ki = 0.5 * (dudx_.get(node, k * nDim_ + i) +
                                   dudx_.get(node, i * nDim_ + k));
        chi_numer += rot_ij * rot_jk * str_ki;
      }
    }
  }

  // JAM: Changes for SWH LowRe
  const NodeKernelTraits::DblType alpha0_star = 0.072 / 3.0;
  const NodeKernelTraits::DblType alpha_inf = 0.52;
  const NodeKernelTraits::DblType alpha0 = 1.0 / 9.0;
  const NodeKernelTraits::DblType Rk = 6.0;
  const NodeKernelTraits::DblType R_beta = 8.0;
  const NodeKernelTraits::DblType Rw = 2.95;
  const NodeKernelTraits::DblType ReT = density * tke / sdr / visc;
  const DblType Rbeta = 8.0;
  const DblType betaStarLowRe =
    betaStar_ * (4.0 / 15.0 + stk::math::pow(ReT / Rbeta, 4.0)) /
    (1.0 + stk::math::pow(ReT / Rbeta, 4.0));
  DblType Dk = betaStarLowRe * density * sdr * tke;

  const DblType chi_omega = stk::math::abs(
    chi_numer / stk::math::pow(0.09 * stk::math::max(sdr, 1.e-8), 3.0));
  const DblType beta =
    0.072 * (1.0 + 70.0 * chi_omega) / (1.0 + 80.0 * chi_omega);

  // JAM: Added for SWH LowRe
  const NodeKernelTraits::DblType alpha_star =
    (alpha0_star + ReT / Rk) / (1.0 + ReT / Rk);
  const NodeKernelTraits::DblType alpha =
    (alpha_inf / alpha_star) * ((alpha0 + ReT / Rw) / (1.0 + ReT / Rw));

  // Pw includes 1/tvisc scaling; tvisc may be zero at a dirichlet low Re
  // approach (clip)
  // JAM: Changes for SWH LowRe, check densities...
  const NodeKernelTraits::DblType Pw =
    alpha * Pk * sdr / stk::math::max(tke, 1.e-12);
  // Production term with appropriate clipping of tvisc
  const DblType Dw = beta * density * sdr * sdr;

  rhs(0) += (Pw - Dw) * dVol;
  lhs(0, 0) += 2.0 * beta * density * sdr * dVol;
}

} // namespace nalu
} // namespace sierra
