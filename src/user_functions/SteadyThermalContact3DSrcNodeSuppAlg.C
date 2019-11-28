// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/SteadyThermalContact3DSrcNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>

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
// SteadyThermalContact3DSrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SteadyThermalContact3DSrcNodeSuppAlg::SteadyThermalContact3DSrcNodeSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    coordinates_(NULL),
    dualNodalVolume_(NULL),
    a_(1.0),
    k_(1.0),
    pi_(std::acos(-1.0))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
SteadyThermalContact3DSrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
SteadyThermalContact3DSrcNodeSuppAlg::node_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double *coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node );
  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];
  rhs[0] += k_/4.0*(2.0*a_*pi_)*(2.0*a_*pi_)*(cos(2.0*a_*pi_*x) + cos(2.0*a_*pi_*y)+ cos(2.0*a_*pi_*z))*dualVolume;
  lhs[0] += 0.0;
}

} // namespace nalu
} // namespace Sierra
