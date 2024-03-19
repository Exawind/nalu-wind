// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/BoussinesqNonIsoEnthalpySrcNodeSuppAlg.h>
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
// BoussinesqNonIsoEnthalpySrcNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
BoussinesqNonIsoEnthalpySrcNodeSuppAlg::BoussinesqNonIsoEnthalpySrcNodeSuppAlg(
  Realm& realm)
  : SupplementalAlgorithm(realm), coordinates_(NULL), dualNodalVolume_(NULL)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<double>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------l
void
BoussinesqNonIsoEnthalpySrcNodeSuppAlg::setup()
{
  // nothing
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
BoussinesqNonIsoEnthalpySrcNodeSuppAlg::node_execute(
  double* /*lhs*/, double* rhs, stk::mesh::Entity node)
{
  // deal with lumped mass matrix
  const double* coords = stk::mesh::field_data(*coordinates_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  const double x = coords[0];
  const double y = coords[1];
  const double z = coords[2];

  const double src =
    (cos(2 * M_PI * z) * sin(2 * M_PI * x) * sin(2 * M_PI * y)) / 2.;

  rhs[0] += src * dualVolume;
}

} // namespace nalu
} // namespace sierra
