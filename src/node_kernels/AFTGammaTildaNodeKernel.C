// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include "node_kernels/AFTGammaTildaNodeKernel.h"
#include "Realm.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace kynema_ugf {

AFTGammaTildaNodeKernel::AFTGammaTildaNodeKernel(
  const stk::mesh::MetaData& meta)
  : NGPNodeKernel<AFTGammaTildaNodeKernel>(),
    nTildaID_(get_field_ordinal(meta, "aft_n_tilda")),
    gammaTildaID_(get_field_ordinal(meta, "aft_gamma_tilda")),
    densityID_(get_field_ordinal(meta, "density")),
    viscosityID_(get_field_ordinal(meta, "viscosity")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dudxID_(get_field_ordinal(meta, "dudx")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),
    nDim_(meta.spatial_dimension())
{
}

void
AFTGammaTildaNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  nTilda_ = fieldMgr.get_field<double>(nTildaID_);
  gammaTilda_ = fieldMgr.get_field<double>(gammaTildaID_);
  density_ = fieldMgr.get_field<double>(densityID_);
  viscosity_ = fieldMgr.get_field<double>(viscosityID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dudx_ = fieldMgr.get_field<double>(dudxID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  freestreamTuPercent_ = realm.get_turb_model_constant(TM_fsti);
}

KOKKOS_FUNCTION
void
AFTGammaTildaNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  const DblType nTilda = nTilda_.get(node, 0);
  const DblType gammaTilda = gammaTilda_.get(node, 0);
  const DblType rho = density_.get(node, 0);
  const DblType mu = viscosity_.get(node, 0);
  const DblType muT = tvisc_.get(node, 0);
  const DblType dualVolume = dualNodalVolume_.get(node, 0);

  DblType strainSquared = 0.0;
  DblType vorticitySquared = 0.0;
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
  const DblType turbulenceReynolds = muT / stk::math::max(mu, 1.0e-16);

  const DblType tuPercent = stk::math::max(freestreamTuPercent_, 1.0e-16);
  const DblType tau = 2.5 * stk::math::tanh(tuPercent / 2.5);
  const DblType nCritical = -8.43 - 2.4 * stk::math::log(tau / 100.0);
  const DblType FonsetOne = nTilda / nCritical;
  const DblType FonsetTwo = stk::math::min(FonsetOne, 2.0);
  const DblType FonsetThree = stk::math::max(
    1.0 - stk::math::pow(turbulenceReynolds / 3.5, 3.0), 0.0);
  const DblType Fonset = stk::math::max(FonsetTwo - FonsetThree, 0.0);
  const DblType fTurb = stk::math::exp(
    -stk::math::pow(turbulenceReynolds / 2.0, 4.0));

  const DblType intermittency = stk::math::exp(gammaTilda);
  const DblType productionCoefficient = 100.0 * rho * strainMagnitude * Fonset;
  const DblType destructionCoefficient =
    0.06 * rho * vorticityMagnitude * fTurb;
  const DblType production = productionCoefficient * (1.0 - intermittency);
  const DblType destruction = destructionCoefficient * (50.0 * intermittency - 1.0);

  rhs(0) += (production - destruction) * dualVolume;
  lhs(0, 0) +=
    intermittency * (productionCoefficient + 50.0 * destructionCoefficient) *
    dualVolume;
}

} // namespace kynema_ugf
} // namespace sierra
