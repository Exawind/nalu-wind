/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradElemAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

struct nodalGradientElem{
private:

  //Bucket and Element Data
  stk::mesh::Bucket & b_;
  MasterElement & meSCS_;
  const double * p_shape_function_;

  //InputFields
  ScalarFieldType & scalarQ_;
  ScalarFieldType & dualNodalVolume_;
  VectorFieldType & coordinates_;

  //OutputFields
  VectorFieldType & dqdx_;

  //Parameters
  const int *lrscv;
  const int nDim_;
  const int numScsIp_;
  const int nodesPerElement_;

  // temporary arrays
  std::vector<double> p_scalarQ;
  std::vector<double> p_dualVolume;
  std::vector<double> p_coordinates;
  std::vector<double> p_scs_areav;

public:
  nodalGradientElem(stk::mesh::Bucket & b, MasterElement & meSCS,
      double * p_shape_function,
      ScalarFieldType & scalarQ, VectorFieldType & dqdx,
      ScalarFieldType & dualNodalVolume, VectorFieldType & coordinates,
      int nDim):
      b_(b),
      meSCS_(meSCS),
      p_shape_function_(p_shape_function),
      scalarQ_(scalarQ),
      dualNodalVolume_(dualNodalVolume),
      coordinates_(coordinates),
      dqdx_(dqdx),
      nDim_(nDim),
      numScsIp_(meSCS_.numIntPoints_),
      nodesPerElement_(meSCS_.nodesPerElement_),
      p_scalarQ(nodesPerElement_),
      p_dualVolume(nodesPerElement_),
      p_coordinates(nodesPerElement_*nDim_),
      p_scs_areav(numScsIp_*nDim_)
  {
    lrscv = meSCS_.adjacentNodes();
  }
  void operator()(stk::mesh::Bucket::size_type elem_offset){
    // get elem
    //===============================================
    // gather nodal data; this is how we do it now..
    //===============================================
    stk::mesh::Entity const * node_rels = b_.begin_nodes(elem_offset);
    const int num_nodes = b_.num_nodes(elem_offset);

    for ( int ni = 0; ni < num_nodes; ++ni ) {
      stk::mesh::Entity node = node_rels[ni];

      const double * coords = stk::mesh::field_data(coordinates_, node );

      // gather scalars
      p_scalarQ[ni]    = *stk::mesh::field_data(scalarQ_, node);
      p_dualVolume[ni] = *stk::mesh::field_data(dualNodalVolume_, node);

      // gather vectors
      const int offSet = ni*nDim_;
      for ( int j=0; j < nDim_; ++j ) {
        p_coordinates[offSet+j] = coords[j];
      }
    }

    // compute geometry
    double scs_error = 0.0;
    meSCS_.determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

    // start assembly
    for ( int ip = 0; ip < numScsIp_; ++ip ) {

      // left and right nodes for this ip
      const int il = lrscv[2*ip];
      const int ir = lrscv[2*ip+1];

      stk::mesh::Entity nodeL = node_rels[il];
      stk::mesh::Entity nodeR = node_rels[ir];

      // pointer to fields to assemble
      double *gradQL = stk::mesh::field_data(dqdx_, nodeL );
      double *gradQR = stk::mesh::field_data(dqdx_, nodeR );

      // interpolate to scs point; operate on saved off ws_field
      double qIp = 0.0;
      const int offSet = ip*nodesPerElement_;
      for ( int ic = 0; ic < nodesPerElement_; ++ic ) {
        qIp += p_shape_function_[offSet+ic]*p_scalarQ[ic];
      }

      // left and right volume
      double inv_volL = 1.0/p_dualVolume[il];
      double inv_volR = 1.0/p_dualVolume[ir];

      // assemble to il/ir
      for ( int j = 0; j < nDim_; ++j ) {
        double fac = qIp*p_scs_areav[ip*nDim_+j];
        gradQL[j] += fac*inv_volL;
        gradQR[j] -= fac*inv_volR;
      }
    }
   }
};

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradElemAlgorithm - Green-Gauss gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradElemAlgorithm::AssembleNodalGradElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  const bool useShifted)
  : Algorithm(realm, part),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    dualNodalVolume_(NULL),
    coordinates_(NULL),
    useShifted_(useShifted)
{
  // extract fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  coordinates_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradElemAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();
  std::vector<double> ws_shape_function;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_) 
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& elem_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    ws_shape_function.resize(numScsIp*nodesPerElement);
    double * p_shape_function = ws_shape_function.data();
    if ( useShifted_ )
      meSCS->shifted_shape_fcn(&p_shape_function[0]);
    else
      meSCS->shape_fcn(&p_shape_function[0]);

    nodalGradientElem nodeGradFunctor(b, *meSCS, p_shape_function, *scalarQ_, *dqdx_, *dualNodalVolume_, *coordinates_, nDim);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      //WARNING: do not thread this functor.  It is not thread-safe because each element scatters to all of its nodes.
      nodeGradFunctor(k);
    }
  }
}




} // namespace nalu
} // namespace Sierra
