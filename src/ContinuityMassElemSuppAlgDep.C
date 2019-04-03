/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <ContinuityMassElemSuppAlgDep.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
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
// ContinuityMassElemSuppAlgDep - CMM (BDF2/BE) for continuity equation (p-dof)
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ContinuityMassElemSuppAlgDep::ContinuityMassElemSuppAlgDep(
   Realm &realm,
   const bool lumpedMass)
  : SupplementalAlgorithm(realm),
    bulkData_(&realm.bulk_data()),
    densityNm1_(NULL),
    densityN_(NULL),
    densityNp1_(NULL),
    coordinates_(NULL),
    dt_(0.0),
    gamma1_(0.0),
    gamma2_(0.0),
    gamma3_(0.0),
    nDim_(realm_.spatialDimension_),
    lumpedMass_(lumpedMass)
{
  // save off fields; shove state N into Nm1 if this is BE
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNm1_ = realm_.number_of_states() == 2 ? &(density->field_of_state(stk::mesh::StateN)) : &(density->field_of_state(stk::mesh::StateNM1));
  densityN_ = &(density->field_of_state(stk::mesh::StateN));
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
}

//--------------------------------------------------------------------------
//-------- elem_resize -----------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityMassElemSuppAlgDep::elem_resize(
  MasterElement */*meSCS*/,
  MasterElement *meSCV)
{
  const int nodesPerElement = meSCV->nodesPerElement_;
  const int numScvIp = meSCV->num_integration_points();

  // resize
  ws_shape_function_.resize(numScvIp*nodesPerElement);
  ws_rhoNp1_.resize(nodesPerElement);
  ws_rhoN_.resize(nodesPerElement);
  ws_rhoNm1_.resize(nodesPerElement);
  ws_coordinates_.resize(nDim_*nodesPerElement);
  ws_scv_volume_.resize(numScvIp);

  // compute shape function
  if ( lumpedMass_ )
    meSCV->shifted_shape_fcn(&ws_shape_function_[0]);
  else
    meSCV->shape_fcn(&ws_shape_function_[0]);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityMassElemSuppAlgDep::setup()
{
  dt_ = realm_.get_time_step();
  gamma1_ = realm_.get_gamma1();
  gamma2_ = realm_.get_gamma2();
  gamma3_ = realm_.get_gamma3(); // gamma3 may be zero
}

//--------------------------------------------------------------------------
//-------- elem_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
ContinuityMassElemSuppAlgDep::elem_execute(
  double */*lhs*/,
  double *rhs,
  stk::mesh::Entity element,
  MasterElement */*meSCS*/,
  MasterElement *meSCV)
{
  const double projTimeScale = dt_/gamma1_;
  
  // pointer to ME methods
  const int *ipNodeMap = meSCV->ipNodeMap();
  const int nodesPerElement = meSCV->nodesPerElement_;
  const int numScvIp = meSCV->num_integration_points();

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
    ws_rhoNm1_[ni] = *stk::mesh::field_data(*densityNm1_, node);
    ws_rhoN_[ni] = *stk::mesh::field_data(*densityN_, node);
    ws_rhoNp1_[ni] = *stk::mesh::field_data(*densityNp1_, node);

    // gather vectors
    const int niNdim = ni*nDim_;
    for ( int i=0; i < nDim_; ++i ) {
      ws_coordinates_[niNdim+i] = coords[i];
    }
  }

  // compute geometry
  double scv_error = 0.0;
  meSCV->determinant(1, &ws_coordinates_[0], &ws_scv_volume_[0], &scv_error);

  for ( int ip = 0; ip < numScvIp; ++ip ) {
      
    // nearest node to ip
    const int nearestNode = ipNodeMap[ip];
    
    // zero out; scalar
    double rhoNm1Scv = 0.0;
    double rhoNScv = 0.0;
    double rhoNp1Scv = 0.0;
      
    const int offSet = ip*nodesPerElement;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      // save off shape function
      const double r = ws_shape_function_[offSet+ic];

      // density
      rhoNm1Scv += r*ws_rhoNm1_[ic];
      rhoNScv += r*ws_rhoN_[ic];
      rhoNp1Scv += r*ws_rhoNp1_[ic];
    }

    // assemble rhs
    const double scV = ws_scv_volume_[ip];
    rhs[nearestNode] += 
      -(gamma1_*rhoNp1Scv + gamma2_*rhoNScv + gamma3_*rhoNm1Scv)*scV/dt_/projTimeScale;
    // manage LHS; n/a
  }
}
  
} // namespace nalu
} // namespace Sierra
