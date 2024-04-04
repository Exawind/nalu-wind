// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <EnthalpyPmrSrcNodeSuppAlg.h>
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
// EnthalpyPmrSrcNodeSuppAlg - base class for algorithm
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
EnthalpyPmrSrcNodeSuppAlg::EnthalpyPmrSrcNodeSuppAlg(Realm& realm)
  : SupplementalAlgorithm(realm), divRadFlux_(NULL), dualNodalVolume_(NULL)
{
  // save off fields
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  divRadFlux_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_radiative_heat_flux");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
EnthalpyPmrSrcNodeSuppAlg::node_execute(
  double* lhs, double* rhs, stk::mesh::Entity node)
{
  // explicit coupling for now...
  const double divQ = *stk::mesh::field_data(*divRadFlux_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);

  rhs[0] -= divQ * dualVolume;
  lhs[0] += 0.0;
}

} // namespace nalu
} // namespace sierra
