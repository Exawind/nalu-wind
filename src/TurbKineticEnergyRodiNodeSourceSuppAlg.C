/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include "TurbKineticEnergyRodiNodeSourceSuppAlg.h"
#include "FieldTypeDef.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "SupplementalAlgorithm.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// TurbKineticEnergyRodiNodeSourceSuppAlg Pb = beta*mu^t/Pr^t gi/Cp dh/dxi
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbKineticEnergyRodiNodeSourceSuppAlg::TurbKineticEnergyRodiNodeSourceSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    dhdx_(NULL),
    specificHeat_(NULL),
    tvisc_(NULL),
    dualNodalVolume_(NULL),
    beta_(realm.solutionOptions_->thermalExpansionCoeff_),
    turbPr_(realm.get_turb_prandtl("enthalpy")),
    nDim_(realm_.meta_data().spatial_dimension())
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  dhdx_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "dhdx");
  specificHeat_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "specific_heat");
  tvisc_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_viscosity");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // extract and size gravity
  gravity_.resize(nDim_);
  gravity_ = realm_.solutionOptions_->gravity_;
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyRodiNodeSourceSuppAlg::setup()
{
  // all set up in constructor
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyRodiNodeSourceSuppAlg::node_execute(
  double */*lhs*/,
  double *rhs,
  stk::mesh::Entity node)
{
  // no lhs contribution; all rhs source term
  const double *dhdx = stk::mesh::field_data(*dhdx_, node );
  const double specificHeat = *stk::mesh::field_data(*specificHeat_, node );
  const double tvisc = *stk::mesh::field_data(*tvisc_, node );
  const double dualNodalVolume = *stk::mesh::field_data(*dualNodalVolume_, node );
  double sum = 0.0;
  for ( int i = 0; i < nDim_; ++i )
    sum += gravity_[i]*dhdx[i];
  rhs[0] += beta_*tvisc/turbPr_*sum/specificHeat*dualNodalVolume;
}

} // namespace nalu
} // namespace Sierra
