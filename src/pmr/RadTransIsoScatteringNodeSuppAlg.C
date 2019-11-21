// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <pmr/RadTransIsoScatteringNodeSuppAlg.h>
#include <pmr/RadiativeTransportEquationSystem.h>
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
// RadTransIsoScatteringNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
RadTransIsoScatteringNodeSuppAlg::RadTransIsoScatteringNodeSuppAlg(
  Realm &realm,
  RadiativeTransportEquationSystem *radEqSystem)
  : SupplementalAlgorithm(realm),
    radEqSystem_(radEqSystem),
    scalarFlux_(NULL),
    scattering_(NULL),
    dualNodalVolume_(NULL),
    invPi_(1.0/std::acos(-1.0))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  scalarFlux_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scalar_flux");
  scattering_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
RadTransIsoScatteringNodeSuppAlg::node_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity node)
{
  // (-beta*G/4/pi)*dVol = 0.0
  const double scalarFlux  = *stk::mesh::field_data(*scalarFlux_, node);
  const double beta = *stk::mesh::field_data(*scattering_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  rhs[0] += beta*scalarFlux/4.0*invPi_*dualVolume;
  lhs[0] += 0.0;
}

} // namespace nalu
} // namespace Sierra
