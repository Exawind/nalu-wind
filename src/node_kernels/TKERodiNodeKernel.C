// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/TKERodiNodeKernel.h"
#include "node_kernels/NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>
#include "utils/StkHelpers.h"
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/Types.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// TKERodiNodeKernel Pb = beta*mu^t/Pr^t gi/Cp dh/dxi
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TKERodiNodeKernel::TKERodiNodeKernel(
  const stk::mesh::MetaData& meta, const SolutionOptions& solnOpts)
  : NGPNodeKernel<TKERodiNodeKernel>(),

    dhdxID_(get_field_ordinal(meta, "dhdx")),
    specificHeatID_(get_field_ordinal(meta, "specific_heat")),
    tviscID_(get_field_ordinal(meta, "turbulent_viscosity")),
    dualNodalVolumeID_(get_field_ordinal(meta, "dual_nodal_volume")),

    turbPr_(0), // Set in setup()
    beta_(solnOpts.thermalExpansionCoeff_),
    nDim_(meta.spatial_dimension())
{
  const std::vector<double>& solnOptsGravity =
    solnOpts.get_gravity_vector(nDim_);
  for (int i = 0; i < nDim_; i++)
    gravity_[i] = solnOptsGravity[i];
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TKERodiNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dhdx_ = fieldMgr.get_field<double>(dhdxID_);
  specificHeat_ = fieldMgr.get_field<double>(specificHeatID_);
  tvisc_ = fieldMgr.get_field<double>(tviscID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  turbPr_ = realm.get_turb_prandtl("enthalpy");
}

//--------------------------------------------------------------------------
//-------- execute ----------------------------------------------------
//--------------------------------------------------------------------------
KOKKOS_FUNCTION
void
TKERodiNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  typedef NodeKernelTraits::DblType Dbl;
  const Dbl dualVolume = dualNodalVolume_.get(node, 0);
  const Dbl specificHeat = specificHeat_.get(node, 0);
  const Dbl tvisc = tvisc_.get(node, 0);

  Dbl sum = 0.0;
  for (int i = 0; i < nDim_; ++i) {
    const Dbl dhdx = dhdx_.get(node, i);
    sum += gravity_[i] * dhdx;
  }
  // no lhs contribution; all rhs source term
  rhs(0) += beta_ * tvisc / turbPr_ * sum / specificHeat * dualVolume;
}

} // namespace nalu
} // namespace sierra
