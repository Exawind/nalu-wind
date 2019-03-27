/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradUElemAlgorithm.h>
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

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradUElemAlgorithm - Green-Gauss gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradUElemAlgorithm::AssembleNodalGradUElemAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  VectorFieldType *vectorQ,
  GenericFieldType *dqdx,
  const bool useShifted)
  : Algorithm(realm, part),
    vectorQ_(vectorQ),
    dqdx_(dqdx),
    useShifted_(useShifted)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradUElemAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // nodal fields to gather; gather everything other than what we are assembling
  std::vector<double> ws_vectorQ;
  std::vector<double> ws_dualVolume;
  std::vector<double> ws_coordinates;

  // geometry related to populate
  std::vector<double> ws_scs_areav;
  std::vector<double> ws_shape_function;

  // ip data
  std::vector<double>qIp(nDim);

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

    // extract master element specifics
    const int nodesPerElement = meSCS->nodesPerElement_;
    const int numScsIp = meSCS->numIntPoints_;
    const int *lrscv = meSCS->adjacentNodes();

    // algorithm related
    ws_vectorQ.resize(nodesPerElement*nDim);
    ws_dualVolume.resize(nodesPerElement);
    ws_coordinates.resize(nodesPerElement*nDim);
    ws_scs_areav.resize(numScsIp*nDim);
    ws_shape_function.resize(numScsIp*nodesPerElement);

    // pointers.
    double *p_vectorQ = &ws_vectorQ[0];
    double *p_dualVolume = &ws_dualVolume[0];
    double *p_coordinates = &ws_coordinates[0];
    double *p_scs_areav = &ws_scs_areav[0];
    double *p_shape_function = &ws_shape_function[0];

    if ( useShifted_ )
      meSCS->shifted_shape_fcn(&p_shape_function[0]);
    else
      meSCS->shape_fcn(&p_shape_function[0]);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      // note: we absolutely need to gather coords since it
      // is required to compute the area vector. however,
      // ws_scalarQ and ws_dualVolume are choices to avoid
      // field data call for interpolation

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        double * coords = stk::mesh::field_data(*coordinates, node);
        double * vectorQ = stk::mesh::field_data(*vectorQ_, node);

        // gather scalars
        p_dualVolume[ni] = *stk::mesh::field_data(*dualNodalVolume, node);

        // gather vectors
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          p_coordinates[offSet+j] = coords[j];
          p_vectorQ[offSet+j] = vectorQ[j];
        }
      }

      // compute geometry
      double scs_error = 0.0;
      meSCS->determinant(1, &p_coordinates[0], &p_scs_areav[0], &scs_error);

      // start assembly
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // left and right nodes for this ip
        const int il = lrscv[2*ip];
        const int ir = lrscv[2*ip+1];

        stk::mesh::Entity nodeL = node_rels[il];
        stk::mesh::Entity nodeR = node_rels[ir];

        // pointer to fields to assemble
        double *gradQL = stk::mesh::field_data(*dqdx_, nodeL);
        double *gradQR = stk::mesh::field_data(*dqdx_, nodeR);

        // interpolate to scs point; operate on saved off ws_field
        for (int j=0; j < nDim; ++j )
          qIp[j] = 0.0;

        const int offSet = ip*nodesPerElement;
        for ( int ic = 0; ic < nodesPerElement; ++ic ) {
          const double r = p_shape_function[offSet+ic];
          for ( int j = 0; j < nDim; ++j ) {
            qIp[j] += r*p_vectorQ[ic*nDim+j];
          }
        }

        // left and right volume
        double inv_volL = 1.0/p_dualVolume[il];
        double inv_volR = 1.0/p_dualVolume[ir];

        // assemble to il/ir
        for ( int i = 0; i < nDim; ++i ) {
          const int row_gradQ = i*nDim;
          const double qip = qIp[i];
          for ( int j = 0; j < nDim; ++j ) {
            double fac = qip*p_scs_areav[ip*nDim+j];
            gradQL[row_gradQ+j] += fac*inv_volL;
            gradQR[row_gradQ+j] -= fac*inv_volR;
          }
        }
      }
    }
  }
}

} // namespace nalu
} // namespace Sierra
