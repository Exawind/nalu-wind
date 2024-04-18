// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKESSTNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SimdInterface.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

TKESSTNodeKernel::TKESSTNodeKernel(
  const stk::mesh::MetaData& meta,
  const FieldManager& manager,
  stk::mesh::PartVector& parts)
  : NGPNodeKernel<TKESSTNodeKernel>(), nDim_(meta.spatial_dimension())
{
  manager.register_field("turbulent_ke", parts);
  manager.register_field("specific_dissipation_rate", parts);
  manager.register_field("density", parts);
  manager.register_field("turbulent_viscosity", parts);
  manager.register_field("dudx", parts);
  manager.register_field("dual_nodal_volume", parts);
}

void
TKESSTNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = *(realm.fieldManager_.get());

  tke_ = fieldMgr.get_ngp_field_ptr<double>("turbulent_ke");
  sdr_ = fieldMgr.get_ngp_field_ptr<double>("specific_dissipation_rate");
  density_ = fieldMgr.get_ngp_field_ptr<double>("density");
  tvisc_ = fieldMgr.get_ngp_field_ptr<double>("turbulent_viscosity");
  dudx_ = fieldMgr.get_ngp_field_ptr<double>("dudx");
  dualNodalVolume_ = fieldMgr.get_ngp_field_ptr<double>("dual_nodal_volume");

  // Update turbulence model constants
  betaStar_ = realm.get_turb_model_constant(TM_betaStar);
  tkeProdLimitRatio_ = realm.get_turb_model_constant(TM_tkeProdLimitRatio);
  tkeAmb_ = realm.get_turb_model_constant(TM_tkeAmb);
  sdrAmb_ = realm.get_turb_model_constant(TM_sdrAmb);
}

KOKKOS_FUNCTION
void
TKESSTNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  using DblType = NodeKernelTraits::DblType;

  // See https://turbmodels.larc.nasa.gov/sst.html for details

  const DblType tke = tke_.get(node, 0);
  const DblType sdr = sdr_.get(node, 0);
  const DblType density = density_.get(node, 0);
  const DblType tvisc = tvisc_.get(node, 0);
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
  const DblType Dk = betaStar_ * density * sdr * tke;

  // Clip production term and prevent Pk from being negative
  Pk = stk::math::min(tkeProdLimitRatio_ * Dk, stk::math::max(Pk, 0.0));

  // SUST source term
  const DblType Dkamb = betaStar_ * density * sdrAmb_ * tkeAmb_;

  rhs(0) += (Pk - Dk + Dkamb) * dVol;
  lhs(0, 0) += betaStar_ * density * sdr * dVol;
}

} // namespace nalu
} // namespace sierra
