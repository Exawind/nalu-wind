// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <EnthalpyLowSpeedCompressibleNodeSuppAlg.h>
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
// EnthalpyLowSpeedCompressibleNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
EnthalpyLowSpeedCompressibleNodeSuppAlg::
  EnthalpyLowSpeedCompressibleNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm),
    pressureN_(NULL),
    pressureNp1_(NULL),
    dualNodalVolume_(NULL),
    dt_(0.0)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  pressureN_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "pressure_old");
  pressureNp1_ =
    meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "pressure");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyLowSpeedCompressibleNodeSuppAlg::setup()
{
  dt_ = realm_.timeIntegrator_->get_time_step();
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyLowSpeedCompressibleNodeSuppAlg::node_execute(
  double* lhs, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double pN = *stk::mesh::field_data(*pressureN_, node);
  const double pNp1 = *stk::mesh::field_data(*pressureNp1_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  rhs[0] += (pNp1 - pN) * dualVolume / dt_;
  lhs[0] += 0.0;
}

} // namespace nalu
} // namespace sierra
