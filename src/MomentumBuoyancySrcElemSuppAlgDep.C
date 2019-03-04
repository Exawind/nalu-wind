/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <MomentumBuoyancySrcElemSuppAlgDep.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>

// master element
#include <master_element/MasterElement.h>

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
// MomentumBuoyancySrcElemSuppAlgDep - CMM for momentum buoyancy (u-dof)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
MomentumBuoyancySrcElemSuppAlgDep::MomentumBuoyancySrcElemSuppAlgDep(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    bulkData_(&realm.bulk_data()),
    densityNp1_(NULL),
    coordinates_(NULL),
    nDim_(realm_.spatialDimension_),
    rhoRef_(0.0),
    useShifted_(false)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  // extract user parameters from solution options
  gravity_.resize(nDim_);
  gravity_ = realm_.solutionOptions_->gravity_;
  rhoRef_ = realm_.solutionOptions_->referenceDensity_;
}

//--------------------------------------------------------------------------
//-------- elem_resize -----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBuoyancySrcElemSuppAlgDep::elem_resize(
  MasterElement */*meSCS*/,
  MasterElement *meSCV)
{
  const int nodesPerElement = meSCV->nodesPerElement_;
  const int numScvIp = meSCV->numIntPoints_;

  // resize
  ws_shape_function_.resize(numScvIp*nodesPerElement);
  ws_rhoNp1_.resize(nodesPerElement);
  ws_coordinates_.resize(nDim_*nodesPerElement);
  ws_scv_volume_.resize(numScvIp);

  // compute shape function
  if ( useShifted_ )
    meSCV->shifted_shape_fcn(&ws_shape_function_[0]);
  else
    meSCV->shape_fcn(&ws_shape_function_[0]);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBuoyancySrcElemSuppAlgDep::setup()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- elem_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
MomentumBuoyancySrcElemSuppAlgDep::elem_execute(
  double * /* lhs */,
  double *rhs,
  stk::mesh::Entity element,
  MasterElement */*meSCS*/,
  MasterElement *meSCV)
{
  // pointer to ME methods
  const int *ipNodeMap = meSCV->ipNodeMap();
  const int nodesPerElement = meSCV->nodesPerElement_;
  const int numScvIp = meSCV->numIntPoints_;

  // gather
  stk::mesh::Entity const *  node_rels = bulkData_->begin_nodes(element);
  int num_nodes = bulkData_->num_nodes(element);

  // sanity check on num nodes
  ThrowAssert( num_nodes == nodesPerElement );

  for ( int ni = 0; ni < num_nodes; ++ni ) {
    stk::mesh::Entity node = node_rels[ni];
    // pointers to real data
    const double * coords =  stk::mesh::field_data(*coordinates_, node);
  
    // gather scalars
    ws_rhoNp1_[ni] = *stk::mesh::field_data(*densityNp1_, node);

    // gather vectors
    const int niNdim = ni*nDim_;
    for ( int j=0; j < nDim_; ++j ) {
      ws_coordinates_[niNdim+j] = coords[j];
    }
  }

  // compute geometry
  double scv_error = 0.0;
  meSCV->determinant(1, &ws_coordinates_[0], &ws_scv_volume_[0], &scv_error);

  for ( int ip = 0; ip < numScvIp; ++ip ) {
      
    // nearest node to ip
    const int nearestNode = ipNodeMap[ip];
    
    // zero out
    double rhoNp1Scv = 0.0;
      
    const int offSet = ip*nodesPerElement;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      // save off shape function
      const double r = ws_shape_function_[offSet+ic];

      // density
      rhoNp1Scv += r*ws_rhoNp1_[ic];
    }

    // assemble rhs
    const double scV = ws_scv_volume_[ip];
    const int nnNdim = nearestNode*nDim_;
    const double fac = (rhoNp1Scv-rhoRef_)*scV;
    for ( int i = 0; i < nDim_; ++i ) {
      rhs[nnNdim+i] += fac*gravity_[i];
    }
    // manage LHS; n/a
  }
}

} // namespace nalu
} // namespace Sierra
