/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <MomentumActuatorSrcNodeSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

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
// MomentumActuatorSrcNodeSuppAlg - actuator line drag source term
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumActuatorSrcNodeSuppAlg::MomentumActuatorSrcNodeSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    actuatorSrc_(NULL),
    actuatorSrcLHS_(NULL),
    dualNodalVolume_(NULL),
    nDim_(1)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  actuatorSrc_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "actuator_source");
  actuatorSrcLHS_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "actuator_source_lhs");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  nDim_ = meta_data.spatial_dimension();
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumActuatorSrcNodeSuppAlg::setup()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- node_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumActuatorSrcNodeSuppAlg::node_execute(
  double *lhs,
  double *rhs,
  stk::mesh::Entity node)
{
  // single point quadrature
  const double *src = stk::mesh::field_data(*actuatorSrc_, node);
  const double dualVolume = *stk::mesh::field_data(*dualNodalVolume_, node);
  const double srcLHS = *stk::mesh::field_data(*actuatorSrcLHS_, node);
 
  const int nDim = nDim_;
  for ( int i = 0; i < nDim; ++i ) {
    rhs[i] += src[i]*dualVolume;
    const int row = i*nDim;
    lhs[row+i] += srcLHS*dualVolume;
  }
}

} // namespace nalu
} // namespace Sierra
