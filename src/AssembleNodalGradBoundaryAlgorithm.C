/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleNodalGradBoundaryAlgorithm.h>
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
// AssembleNodalGradBoundaryAlgorithm - adds in boundary contribution
//                                      for elem/edge proj nodal gradient
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradBoundaryAlgorithm::AssembleNodalGradBoundaryAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  ScalarFieldType *scalarQ,
  VectorFieldType *dqdx,
  const bool useShifted)
  : Algorithm(realm, part),
    scalarQ_(scalarQ),
    dqdx_(dqdx),
    useShifted_(useShifted)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradBoundaryAlgorithm::execute()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract fields
  GenericFieldType *exposedAreaVec = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // nodal fields to gather; gather everything other than what we are assembling
  std::vector<double> ws_scalarQ;

  // geometry related to populate
  std::vector<double> ws_shape_function;

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();

    // extract master element
    MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());

    // extract master element specifics
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsIp = meFC->num_integration_points();
    const int *ipNodeMap = meFC->ipNodeMap();

    // algorithm related
    ws_scalarQ.resize(nodesPerFace);
    ws_shape_function.resize(numScsIp*nodesPerFace);

    // pointers
    double *p_scalarQ = ws_scalarQ.data();
    double *p_shape_function = ws_shape_function.data();

    if ( useShifted_ )
      meFC->shifted_shape_fcn(&p_shape_function[0]);
    else
      meFC->shape_fcn(&p_shape_function[0]);

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec, b, k);

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * face_node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerFace );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather scalars
        ws_scalarQ[ni] = *stk::mesh::field_data(*scalarQ_, node);
      }

      // start assembly
      for ( int ip = 0; ip < numScsIp; ++ip ) {

        // nearest node
        const int nn = ipNodeMap[ip];

        stk::mesh::Entity nodeNN = face_node_rels[nn];

        // pointer to fields to assemble
        double *gradQNN = stk::mesh::field_data(*dqdx_, nodeNN);

        // suplemental
        double volNN = *stk::mesh::field_data(*dualNodalVolume, nodeNN);

        // interpolate to scs point; operate on saved off ws_field
        double qIp = 0.0;
        const int offSet = ip*nodesPerFace;
        for ( int ic = 0; ic < nodesPerFace; ++ic ) {
          qIp += p_shape_function[offSet+ic]*p_scalarQ[ic];
        }

        // nearest node volume
        double inv_volNN = 1.0/volNN;

        // assemble to nearest node
        for ( int j = 0; j < nDim; ++j ) {
          double fac = qIp*areaVec[ip*nDim+j];
          gradQNN[j] += fac*inv_volNN;
        }
      }
    }
  }
}

} // namespace nalu
} // namespace Sierra
