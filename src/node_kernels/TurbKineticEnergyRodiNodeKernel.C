/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "node_kernels/TurbKineticEnergyRodiNodeKernel.h"
#include "node_kernels/NodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

// stk_mesh/base/fem
#include <stk_mesh/base/MetaData.hpp>
#include "utils/StkHelpers.h"
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbKineticEnergyRodiNodeKernel Pb = beta*mu^t/Pr^t gi/Cp dh/dxi
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbKineticEnergyRodiNodeKernel::TurbKineticEnergyRodiNodeKernel(
    const stk::mesh::MetaData& meta,
    const SolutionOptions& solnOpts) 
  : NGPNodeKernel<TurbKineticEnergyRodiNodeKernel>(),

    dhdxID_             (get_field_ordinal(meta, "dhdx")),
    specificHeatID_     (get_field_ordinal(meta, "specific_heat")),
    tviscID_            (get_field_ordinal(meta, "turbulent_viscosity")),
    dualNodalVolumeID_  (get_field_ordinal(meta, "dual_nodal_volume")),

    turbPr_             (0), // Set in setup()
    beta_               (solnOpts.thermalExpansionCoeff_),
    nDim_               (meta.spatial_dimension()) 
{
  const std::vector<double>& solnOptsGravity = solnOpts.get_gravity_vector(nDim_);
  for (int i = 0; i < nDim_; i++)
    gravity_[i] = solnOptsGravity[i];
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyRodiNodeKernel::setup(Realm &realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dhdx_            = fieldMgr.get_field<double>(dhdxID_);
  specificHeat_    = fieldMgr.get_field<double>(specificHeatID_);
  tvisc_           = fieldMgr.get_field<double>(tviscID_);
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);

  turbPr_ = realm.get_turb_prandtl("enthalpy");
}

//--------------------------------------------------------------------------
//-------- execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyRodiNodeKernel::execute(
  NodeKernelTraits::LhsType& /*lhs*/,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  typedef NodeKernelTraits::DblType Dbl;
  const Dbl dualVolume   = dualNodalVolume_.get(node, 0);
  const Dbl specificHeat = specificHeat_.   get(node, 0);
  const Dbl tvisc        = tvisc_.          get(node, 0);

  Dbl sum = 0.0;
  for ( int i = 0; i < nDim_; ++i ) {
    const Dbl dhdx = dhdx_.get(node, i);
    sum += gravity_[i]*dhdx;
  }
  // no lhs contribution; all rhs source term
  rhs(0) += beta_*tvisc/turbPr_*sum/specificHeat*dualVolume;
}

} // namespace nalu
} // namespace Sierra
