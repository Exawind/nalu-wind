/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumEdgeABLTopBC.h>
#include <SolverAlgorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleMomentumEdgeABLTopBC - Top boundary condition for inflow/outflow
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumEdgeABLTopBC::AssembleMomentumEdgeABLTopBC(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem, std::vector<int>& grid_dims)
  : SolverAlgorithm(realm, part, eqSystem),
  imax_(grid_dims[0]), jmax_(grid_dims[1]), kmax_(grid_dims[2]),
  sampleVel_(imax_*jmax_)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  velocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  bcVelocity_ = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "top_velocity_bc");
  density_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::initialize_connectivity()
{
  eqSystem_->linsys_->buildDirichletNodeGraph(partVec_);
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumEdgeABLTopBC::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // space for LHS/RHS; nodesPerFace*nDim*nodesPerFace*nDim and nodesPerFace*nDim

  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get node
      stk::mesh::Entity node = b[k];

      stk::mesh::EntityId nodeID = bulk_data.identifier(node);
      stk::mesh::EntityId nodeTmp = nodeID -1;
      int iz = (nodeTmp)/(imax_*jmax_);
      nodeTmp %= imax_*jmax_;
      int iy = (nodeTmp/imax_);
      int ix = nodeTmp % imax_;

      stk::mesh::EntityId nodeIDx = bulk_data.identifier(node);
      stk::mesh::EntityId nodeID1 = (iz-20)*imax_*jmax_ + iy*imax_ + ix + 1;
      stk::mesh::EntityId nodeID2 = (iz-1 )*imax_*jmax_ + iy*imax_ + ix + 1;
//      std::cerr << nodeIDx << " " << nodeID1 << " " <<
//       iz << " " << iy << " " << ix << std::endl;
//    stk::mesh::EntityId nodeID1 = nodeID - 10*imax_*jmax_;

      stk::mesh::Entity node1 = bulk_data.get_entity(stk::topology::NODE_RANK,nodeID1);
      stk::mesh::Entity node2 = bulk_data.get_entity(stk::topology::NODE_RANK,nodeID2);

      double *uNp1 = stk::mesh::field_data(velocityNp1, node1);
      double *uBC = stk::mesh::field_data(*bcVelocity_, node );
      double *uTop = stk::mesh::field_data(velocityNp1, node );
      double *uTopm= stk::mesh::field_data(velocityNp1, node2);

      uBC[0] = uNp1[0];
      uBC[1] = uNp1[1];
      uBC[2] = uNp1[2];
//      uTop[0] = uNp1[0];
//      uTop[1] = uNp1[1];
//      uTop[2] = uNp1[2];
//      uTopm[0] = uNp1[0];
//      uTopm[1] = uNp1[1];
//      uTopm[2] = uNp1[2];

      //sampleVel_[iy*imax_+ix] = uNp1[2];
    }
  }

  eqSystem_->linsys_->applyDirichletBCs(
  velocity_,
  bcVelocity_,
  partVec_,
  0,
  2);

}


} // namespace nalu
} // namespace Sierra
