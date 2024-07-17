// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <ContinuityLowSpeedCompressibleNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>

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
// ContinuityLowSpeedCompressibleNodeSuppAlg - add LHS for d/dp(rho)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ContinuityLowSpeedCompressibleNodeSuppAlg::
  ContinuityLowSpeedCompressibleNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    densityNp1_(NULL),
    pressure_(NULL),
    dualNodalVolume_(NULL),
    dt_(0.0),
    gamma1_(1.0)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  ScalarFieldType* density =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  pressure_ = meta_data.get_field<double>(stk::topology::NODE_RANK, "pressure");
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityLowSpeedCompressibleNodeSuppAlg::setup()
{
  gamma1_ = realm_.get_gamma1();
  dt_ = realm_.timeIntegrator_->get_time_step();
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityLowSpeedCompressibleNodeSuppAlg::node_execute(
  double* lhs, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double projTimeScale = dt_ / gamma1_;
  const double rhoNp1 = *stk::mesh::field_data(*densityNp1_, node);
  const double pressure = *stk::mesh::field_data(*pressure_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  rhs[0] +=
    0.0; // no RHS - already provided in standard density time derivative
  lhs[0] += rhoNp1 / pressure * dualVolume / dt_ / projTimeScale;
}

} // namespace nalu
} // namespace sierra
