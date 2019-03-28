/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <ComputeGeometryAlgorithmDriver.h>

#include <Algorithm.h>
#include <AlgorithmDriver.h>
#include <FieldTypeDef.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>
#include <Realm.h>
#include <SolutionOptions.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// ComputeGeometryAlgorithmDriver - Drives nodal grad algorithms
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeGeometryAlgorithmDriver::ComputeGeometryAlgorithmDriver(
  Realm &realm)
  : AlgorithmDriver(realm)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeGeometryAlgorithmDriver::pre_work()
{
  // meta and bulk data
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // extract field that is always germane
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // define some common selectors
  stk::mesh::Selector s_locally_owned = meta_data.locally_owned_part();

  //====================================================
  // Initialize nodal volume and area vector to zero
  //====================================================

  // nodal fields first
  stk::mesh::Selector s_all_vol
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectField(*dualNodalVolume);
  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_vol );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length   = b.size();
    double * nv = stk::mesh::field_data( *dualNodalVolume, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      nv[k] = 0.0;
    }
  }

  // handle case for realm using edge-based
  if ( realm_.realmUsesEdges_ ) {

    VectorFieldType *edgeAreaVec = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");

    // edge fields second
    stk::mesh::Selector s_all_area
      = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
      &stk::mesh::selectField(*edgeAreaVec);
    stk::mesh::BucketVector const& edge_buckets =
      realm_.get_buckets( stk::topology::EDGE_RANK, s_all_area );
    for ( stk::mesh::BucketVector::const_iterator ib = edge_buckets.begin();
          ib != edge_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      const stk::mesh::Bucket::size_type length   = b.size();
      double * av = stk::mesh::field_data(*edgeAreaVec, b);
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        const int offSet = k*nDim;
        for ( int j = 0; j < nDim; ++j )
          av[offSet+j] = 0.0;
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- post_work -------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeGeometryAlgorithmDriver::post_work()
{

  // meta and bulk data
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // extract field always germane
  ScalarFieldType *dualNodalVolume = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "dual_nodal_volume");

  // handle case for realm using edge-based
  if ( realm_.realmUsesEdges_ ) {
    VectorFieldType *edgeAreaVec = meta_data.get_field<VectorFieldType>(stk::topology::EDGE_RANK, "edge_area_vector");
    stk::mesh::parallel_sum(bulk_data, {dualNodalVolume, edgeAreaVec});
  }
  else {
    stk::mesh::parallel_sum(bulk_data, {dualNodalVolume});
  }

  // deal with parallel

  if ( realm_.hasPeriodic_) {
    const unsigned fieldSize = 1;
    realm_.periodic_field_update(dualNodalVolume, fieldSize);
  }

  if ( realm_.checkJacobians_ ) {
    check_jacobians();
  }
}

//--------------------------------------------------------------------------
//-------- check_jacobians -------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeGeometryAlgorithmDriver::check_jacobians()
{
  bool badElemFound = false;

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const int nDim = meta_data.spatial_dimension();

  // fields and future ws 
  VectorFieldType *coordinates 
    = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  std::vector<double> ws_coordinates;
  std::vector<double> ws_scv_volume;
  
  // extract target parts
  std::vector<std::string> targetNames = realm_.get_physics_target_names();
  
  for (size_t itarget=0; itarget < targetNames.size(); ++itarget) {
  
    std::vector<stk::mesh::EntityId> localBadElemVec;
    
    stk::mesh::Part *targetPart = meta_data.get_part(targetNames[itarget]);

    // define some common selectors
    stk::mesh::Selector s_locally_owned = meta_data.locally_owned_part() & stk::mesh::Selector(*targetPart);

    stk::mesh::BucketVector const& elem_buckets =
      realm_.get_buckets( stk::topology::ELEM_RANK, s_locally_owned );
    for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
          ib != elem_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      const stk::mesh::Bucket::size_type length   = b.size();
      
      MasterElement *meSCV = sierra::nalu::MasterElementRepo::get_volume_master_element(b.topology());

      // extract master element specifics
      const int nodesPerElement = meSCV->nodesPerElement_;
      const int numScvIp = meSCV->num_integration_points();

      // resize
      ws_coordinates.resize(nodesPerElement*nDim);
      ws_scv_volume.resize(numScvIp);

      double *p_coordinates = &ws_coordinates[0];
      double *p_scv_volume = &ws_scv_volume[0];
      
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

        stk::mesh::Entity const *  node_rels = b.begin_nodes(k);
        int num_nodes = b.num_nodes(k);
        
        for ( int ni = 0; ni < num_nodes; ++ni ) {
          stk::mesh::Entity node = node_rels[ni];
          const double * coords = stk::mesh::field_data(*coordinates, node );
          const int niNdim = ni*nDim;
          for ( int j=0; j < nDim; ++j ) {
            p_coordinates[niNdim+j] = coords[j];
          }
        }
        
        double scv_error = 0.0;
        meSCV->determinant(1, &p_coordinates[0], &p_scv_volume[0], &scv_error);
        
        bool localBadElem = false;
        for ( int ip = 0; ip < numScvIp; ++ip ) {
          if ( p_scv_volume[ip] < 0.0 )
            localBadElem = true;
        }
        
        if ( localBadElem == true ) {
          localBadElemVec.push_back(bulk_data.identifier(b[k]));
          badElemFound = true;
        }
      }
      
    }
    
    // report
    if ( localBadElemVec.size() > 0 ) {
      NaluEnv::self().naluOutput() << "ComputeGeometryAlgorithmDriver::check_jacobians() ERROR on block " 
                                   << targetPart->name() << std::endl;
      for ( size_t n = 0; n < localBadElemVec.size(); ++n ) 
        NaluEnv::self().naluOutput() << " Negative Jacobian found in Element global Id: " << localBadElemVec[n] << std::endl;
    } 
  }

  // no need to proceed
  if ( badElemFound )
    throw std::runtime_error("ComputeGeometryAlgorithmDriver::check_jacobians() Error");
}
  
} // namespace nalu
} // namespace Sierra
