/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeGeometryInteriorAlgorithm.h>

#include <Realm.h>
#include <FieldTypeDef.h>
#include <master_element/MasterElement.h>
#include "master_element/MasterElementFactory.h"

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_topo
#include <stk_topology/topology.hpp>

// basic c++
#include <vector>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeGeometryInteriorAlgorithm - base class equation system
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeGeometryInteriorAlgorithm::ComputeGeometryInteriorAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    assembleEdgeAreaVec_(realm_.realmUsesEdges_)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeGeometryInteriorAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract field always germane
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");
  VectorFieldType *coordinates = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
 
  // setup for buckets; union parts and ask for locally owned
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    & stk::mesh::selectUnion(partVec_)  
    & !(realm_.get_inactive_selector());

  stk::mesh::BucketVector const& element_buckets =
    realm_.get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );

  //===========================================================
  // nodal volume assembly
  //===========================================================
  for ( stk::mesh::BucketVector::const_iterator ib = element_buckets.begin();
        ib != element_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // extract master element
    MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCV->nodesPerElement_;
    const int numScvIp = meSCV->numIntPoints_;
    const int *ipNodeMap = meSCV->ipNodeMap();

    // define scratch field
    std::vector<double > ws_coordinates(nodesPerElement*nDim);
    std::vector<double > ws_scv_volume(numScvIp);

    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const * node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert( num_nodes == nodesPerElement );

      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = node_rels[ni];
        double * coords = stk::mesh::field_data(*coordinates, node);
        const int offSet = ni*nDim;
        for ( int j=0; j < nDim; ++j ) {
          ws_coordinates[offSet+j] = coords[j];
        }
      }

      // compute integration point volume
      double scv_error = 0.0;
      meSCV->determinant(1, &ws_coordinates[0], &ws_scv_volume[0], &scv_error);

      // assemble dual volume while scattering ip volume
      for ( int ip = 0; ip < numScvIp; ++ip ) {
        // nearest node for this ip
        const int nn = ipNodeMap[ip];
        stk::mesh::Entity node = node_rels[nn];
        double * dualcv = stk::mesh::field_data(*dualNodalVolume, node);
        // augment nodal dual volume
        *dualcv += ws_scv_volume[ip];
      }
    }
  }

  //===========================================================
  // Edge-area
  //===========================================================
  if ( assembleEdgeAreaVec_ ) {

    VectorFieldType *edgeAreaVec = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");

    for ( stk::mesh::BucketVector::const_iterator ib = element_buckets.begin();
          ib != element_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;

      // extract master element
      MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());

      // extract master element specifics
      const int nodesPerElement = meSCS->nodesPerElement_;
      const int numScsIp = meSCS->numIntPoints_;
      const int *lrscv = meSCS->adjacentNodes();
      const int *scsIpEdgeOrd = meSCS->scsIpEdgeOrd();

      // define scratch field
      std::vector<double > ws_coordinates(nodesPerElement*nDim);
      std::vector<double > ws_scs_areav(numScsIp*nDim);
  
      const stk::mesh::Bucket::size_type length   = b.size();

      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

        // Use node Entity because we'll need to call BulkData::identifier(.).
        stk::mesh::Entity const * elem_node_rels = b.begin_nodes(k);

        //===============================================
        // gather nodal data; this is how we do it now..
        //===============================================
        int num_nodes = b.num_nodes(k);
        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = elem_node_rels[ni];
          double * coords = stk::mesh::field_data(*coordinates, node);
          const int offSet = ni*nDim;
          for ( int j=0; j < nDim; ++j ) {
            ws_coordinates[offSet+j] = coords[j];
          }
        }

        // compute scs integration point areavec
        double scs_error = 0.0;
        meSCS->determinant(1, &ws_coordinates[0], &ws_scs_areav[0], &scs_error);

        // extract edge connectivity
        stk::mesh::Entity const * elem_edge_rels = b.begin_edges(k);
                        
        for ( int ip = 0; ip < numScsIp; ++ip ) {
          
          // for this ip, extract the local edge ordinal
          const int nedge = scsIpEdgeOrd[ip];
          
          // get edge and area_vector
          stk::mesh::Entity edge = elem_edge_rels[nedge];
          ThrowAssertMsg(bulk_data.is_valid(edge),"Error!  Invalid edge returned from element relations to edges!");
          
          double * av = stk::mesh::field_data(*edgeAreaVec, edge );
          
          // extract edge->node relations
          stk::mesh::Entity const * edge_node_rels = bulk_data.begin_nodes(edge);
          ThrowAssert( 2 == bulk_data.num_nodes(edge) );

          // work towards "sign" convention

          // extract a local node; choose to pick L and follow it through
          const int iloc_L = lrscv[2*ip];

          // get global identifiers for nodes Left and Right from the element
          const size_t iglob_Lelem = bulk_data.identifier(elem_node_rels[iloc_L]);
          const size_t iglob_Ledge = bulk_data.identifier(edge_node_rels[0]);

          // determine the sign value for area vector; if Left node is the same,
          // then the element and edge relations are aligned
          const double sign = ( iglob_Lelem == iglob_Ledge ) ? 1.0 : -1.0;

          const int offSet = ip*nDim;
          for ( int j = 0; j < nDim; ++j ) {
            av[j] += ws_scs_areav[offSet+j]*sign;
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeGeometryInteriorAlgorithm::~ComputeGeometryInteriorAlgorithm()
{
  // does nothing
}



} // namespace nalu
} // namespace Sierra
