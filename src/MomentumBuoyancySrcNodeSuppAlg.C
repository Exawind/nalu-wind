// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <MomentumBuoyancySrcNodeSuppAlg.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <SupplementalAlgorithm.h>

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// MomentumBuoyancySrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumBuoyancySrcNodeSuppAlg::MomentumBuoyancySrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    densityNp1_(NULL),
    dualNodalVolume_(NULL),
    nDim_(1),
    rhoRef_(0.0)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  ScalarFieldType* density =
    meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");

  rhoRefIsHydroDens_ = realm_.solutionOptions_->rho_ref_to_hydrostatic_rho_;

  if (rhoRefIsHydroDens_) {
    ScalarFieldType* hydroDensity = meta_data.get_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "hydrostatic_density");
    hydroDensityNp1_ = &(hydroDensity->field_of_state(stk::mesh::StateNone));
  }
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));

  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  nDim_ = meta_data.spatial_dimension();
  gravity_.resize(nDim_);

  // extract user parameters from solution options
  gravity_ = realm_.solutionOptions_->gravity_;
  rhoRef_ = realm_.solutionOptions_->referenceDensity_;
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBuoyancySrcNodeSuppAlg::setup()
{
  // all set up in constructor
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBuoyancySrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // rhs+=(rho-rhoRef)*gi
  // later, may choose to assemble buoyancy to scv ips: Nip_k*rho_k
  const double rhoNp1 = *stk::mesh::field_data(*densityNp1_, node);
  const double hydroRhoNp1 = rhoRefIsHydroDens_
                               ? *stk::mesh::field_data(*hydroDensityNp1_, node)
                               : rhoNp1;
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  const double fac = !rhoRefIsHydroDens_ ? (rhoNp1 - rhoRef_) * dualVolume
                                         : (rhoNp1 - hydroRhoNp1) * dualVolume;
  const int nDim = nDim_;
  for (int i = 0; i < nDim; ++i) {
    rhs[i] += fac * gravity_[i];
  }
}

} // namespace nalu
} // namespace sierra
