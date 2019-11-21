// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <pmr/RadTransBlackBodyNodeSuppAlg.h>
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
// RadTransBlackBodyNodeSuppAlg - base class for algorithm
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
RadTransBlackBodyNodeSuppAlg::RadTransBlackBodyNodeSuppAlg(
  Realm &realm,
  RadiativeTransportEquationSystem *radEqSystem)
  : SupplementalAlgorithm(realm),
    radEqSystem_(radEqSystem),
    intensity_(NULL),
    absorption_(NULL),
    scattering_(NULL),
    radiationSource_(NULL),
    dualNodalVolume_(NULL)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  absorption_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "absorption_coefficient");
  scattering_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "scattering_coefficient");
  radiationSource_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "radiation_source");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
RadTransBlackBodyNodeSuppAlg::setup()
{
  intensity_ = radEqSystem_->get_intensity();
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
RadTransBlackBodyNodeSuppAlg::node_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity node)
{
  // ((mu+beta)*I - mu*sigma*T^4/pi)*dVol = 0.0
  const double intensity  = *stk::mesh::field_data(*intensity_, node);
  const double mu = *stk::mesh::field_data(*absorption_, node);
  const double beta = *stk::mesh::field_data(*scattering_, node);
  const double radiationSource = *stk::mesh::field_data(*radiationSource_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  rhs[0] -= ((mu+beta)*intensity - radiationSource)*dualVolume;
  lhs[0] += (mu+beta)*dualVolume;
}

} // namespace nalu
} // namespace Sierra
