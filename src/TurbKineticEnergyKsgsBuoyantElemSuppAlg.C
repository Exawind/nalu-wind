/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <TurbKineticEnergyKsgsBuoyantElemSuppAlg.h>
#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>
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
// TurbKineticEnergyKsgsBuoyantElemSuppAlg - CMM (BDF2) for scalar equation
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
TurbKineticEnergyKsgsBuoyantElemSuppAlg::TurbKineticEnergyKsgsBuoyantElemSuppAlg(
  Realm &realm)
  : SupplementalAlgorithm(realm),
    bulkData_(&realm.bulk_data()),
    tkeNp1_(NULL),
    densityNp1_(NULL),
    dualNodalVolume_(NULL),
    coordinates_(NULL),
    CbTwo_(realm.get_turb_model_constant(TM_CbTwo)),
    nDim_(realm_.meta_data().spatial_dimension())
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  ScalarFieldType *tke = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  tkeNp1_ = &(tke->field_of_state(stk::mesh::StateNP1));
  ScalarFieldType *density = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  densityNp1_ = &(density->field_of_state(stk::mesh::StateNP1));
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // fixed size; for sake of cross product make gravity and density gradient size 3
  ws_gravity_.resize(3,0.0);
  ws_drhodx_.resize(3,0.0);

  // extract gravity from solution options; may not be of size 3; copy into class
  std::vector<double> soGravity = realm_.solutionOptions_->gravity_;
  for ( size_t k = 0; k < soGravity.size(); ++k )
    ws_gravity_[k] = soGravity[k];
}

//--------------------------------------------------------------------------
//-------- elem_resize -----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyKsgsBuoyantElemSuppAlg::elem_resize(
  MasterElement */*meSCS*/,
  MasterElement *meSCV)
{
  const int nodesPerElement = meSCV->nodesPerElement_;
  const int numScvIp = meSCV->num_integration_points();

  // resize
  ws_dndx_.resize(nDim_*numScvIp*nodesPerElement);
  ws_deriv_.resize(nDim_*numScvIp*nodesPerElement);
  ws_det_j_.resize(numScvIp);
  ws_shape_function_.resize(numScvIp*nodesPerElement);
 
  ws_tkeNp1_.resize(nodesPerElement);
  ws_rhoNp1_.resize(nodesPerElement);
  ws_dualNodalVolume_.resize(nodesPerElement);
  ws_coordinates_.resize(nDim_*nodesPerElement);
  ws_scv_volume_.resize(numScvIp);
  
  // compute shape function
  meSCV->shape_fcn(&ws_shape_function_[0]);
}

//--------------------------------------------------------------------------
//-------- setup -----------------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyKsgsBuoyantElemSuppAlg::setup()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- elem_execute ----------------------------------------------------
//--------------------------------------------------------------------------
void
TurbKineticEnergyKsgsBuoyantElemSuppAlg::elem_execute(
  double */*lhs*/,
  double *rhs,
  stk::mesh::Entity element,
  MasterElement */*meSCS*/,
  MasterElement *meSCV)
{
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
    ws_tkeNp1_[ni] = *stk::mesh::field_data(*tkeNp1_, node);
    ws_rhoNp1_[ni] = *stk::mesh::field_data(*densityNp1_, node);
    ws_dualNodalVolume_[ni] = *stk::mesh::field_data(*dualNodalVolume_, node);

    // gather vectors
    const int niNdim = ni*nDim_;
    for ( int i=0; i < nDim_; ++i ) {
      ws_coordinates_[niNdim+i] = coords[i];
    }
  }

  // compute geometry
  double scv_error = 0.0;
  meSCV->determinant(1, &ws_coordinates_[0], &ws_scv_volume_[0], &scv_error);

  // compute dndx
  meSCV->grad_op(1, &ws_coordinates_[0], &ws_dndx_[0], &ws_deriv_[0], &ws_det_j_[0], &scv_error);

  for ( int ip = 0; ip < numScvIp; ++ip ) {
      
    // nearest node to ip
    const int nearestNode = ipNodeMap[ip];
    
    // extract filter based on nearest dual volume
    const double filter = std::pow(ws_dualNodalVolume_[nearestNode], 1.0/nDim_);

    // zero out vector
    for ( int i = 0; i < nDim_; ++i )
      ws_drhodx_[i] = 0.0;
    
    // compute density gradient and tke at SCV
    double tkeScv = 0.0;
    const int offSet = ip*nodesPerElement;
    for ( int ic = 0; ic < nodesPerElement; ++ic ) {
      tkeScv += ws_shape_function_[offSet+ic]*ws_tkeNp1_[ic];
      // compute scv derivative
      const int offSetDnDx = nDim_*nodesPerElement*ip + ic*nDim_;
      const double rhoIC = ws_rhoNp1_[ic];
      for ( int j = 0; j < nDim_; ++j ) {
        const double dn = ws_dndx_[offSetDnDx+j];
        ws_drhodx_[j] += rhoIC*dn;
      }
    }

    // dot product; gradRho(dot)g
    double dotProduct = 0.0;
    for ( int i = 0; i < nDim_; ++i )
      dotProduct += ws_drhodx_[i]*ws_gravity_[i];

    // cross product magnitude; | gradRho x g |
    const double crossProductMag = cross_product_magnitude();
    
    // assemble rhs; no chance for stable LHS
    const double scV = ws_scv_volume_[ip];
    rhs[nearestNode] += CbTwo_*filter*std::sqrt(tkeScv)*(crossProductMag - dotProduct)*scV;
  }
}

//--------------------------------------------------------------------------
//-------- cross_product_magnitude -----------------------------------------
//--------------------------------------------------------------------------
double
TurbKineticEnergyKsgsBuoyantElemSuppAlg::cross_product_magnitude()
{
  const double crossX =   ws_drhodx_[1]*ws_gravity_[2] - ws_drhodx_[2]*ws_gravity_[1];
  const double crossY = -(ws_drhodx_[0]*ws_gravity_[2] - ws_drhodx_[2]*ws_gravity_[0]);
  const double crossZ = ws_drhodx_[0]*ws_gravity_[1] - ws_drhodx_[1]*ws_gravity_[0];
  return std::sqrt(crossX*crossX + crossY*crossY + crossZ*crossZ);
}  
} // namespace nalu
} // namespace Sierra
