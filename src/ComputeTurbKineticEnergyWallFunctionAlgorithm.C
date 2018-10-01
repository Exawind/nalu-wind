/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeTurbKineticEnergyWallFunctionAlgorithm.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>

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
// ComputeTurbKineticEnergyWallFunctionAlgorithm - utau at wall bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeTurbKineticEnergyWallFunctionAlgorithm::ComputeTurbKineticEnergyWallFunctionAlgorithm(
  Realm &realm,
  stk::mesh::Part *part)
  : Algorithm(realm, part),
    cMu_(realm.get_turb_model_constant(TM_cMu))
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  turbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "turbulent_ke");
  bcTurbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tke_bc");
  bcAssembledTurbKineticEnergy_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "wall_model_tke_bc");
  assembledWallArea_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_wf");
  wallFrictionVelocityBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "wall_friction_velocity_bip");
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeTurbKineticEnergyWallFunctionAlgorithm::~ComputeTurbKineticEnergyWallFunctionAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeTurbKineticEnergyWallFunctionAlgorithm::execute()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // zero out assembled nodal quantities
  zero_nodal_fields();

  // define some common selectors
  stk::mesh::Selector s_locally_owned_union = meta_data.locally_owned_part()
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& face_buckets =
    realm_.get_buckets( meta_data.side_rank(), s_locally_owned_union );
  for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
        ib != face_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;

    // face master element; only need the face topo
    const int nodesPerFace = b.topology().num_nodes();
    const stk::mesh::Bucket::size_type length   = b.size();

    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {

      // get face
      stk::mesh::Entity face = b[k];

      // get relations to nodes
      stk::mesh::Entity const * face_node_rels = bulk_data.begin_nodes(face);

      // pointer to face data
      const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
      const double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);

      // loop over face nodes
      for ( int ip = 0; ip < nodesPerFace; ++ip ) {

        const int offSetAveraVec = ip*nDim;

        // nearest node
        stk::mesh::Entity node = face_node_rels[ip];

        // compute aMag
        double aMag = 0.0;
        for ( int j = 0; j < nDim; ++j ) {
          const double axj = areaVec[offSetAveraVec+j];
          aMag += axj*axj;
        }
        aMag = std::sqrt(aMag);

        // extract utau and compute wall valu for tke
        const double utau = wallFrictionVelocityBip[ip];
        const double tkeBip = utau*utau/std::sqrt(cMu_);
        // assemble to nodal quantities
        double * bcAssembledTurbKineticEnergy = stk::mesh::field_data(*bcAssembledTurbKineticEnergy_, node );
        *bcAssembledTurbKineticEnergy += aMag*tkeBip;
      }
    }
  }

  // parallel assemble and normalize
  normalize_nodal_fields();
}

//--------------------------------------------------------------------------
//-------- zero_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeTurbKineticEnergyWallFunctionAlgorithm::zero_nodal_fields()
{

  stk::mesh::MetaData & meta_data = realm_.meta_data();

  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    double * bcAssembledTurbKineticEnergy = stk::mesh::field_data(*bcAssembledTurbKineticEnergy_, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      bcAssembledTurbKineticEnergy[k] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------
//-------- normalize_nodal_fields -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeTurbKineticEnergyWallFunctionAlgorithm::normalize_nodal_fields()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // deal with state
  ScalarFieldType &tkeNp1 = turbKineticEnergy_->field_of_state(stk::mesh::StateNP1);

  stk::mesh::parallel_sum(bulk_data, {bcAssembledTurbKineticEnergy_});

  // periodic assemble
  if ( realm_.hasPeriodic_) {
    const unsigned fieldSize = 1;
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(bcAssembledTurbKineticEnergy_, fieldSize, bypassFieldCheck);
  }

  // normalize
  stk::mesh::Selector s_all_nodes
    = (meta_data.locally_owned_part() | meta_data.globally_shared_part())
    &stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets( stk::topology::NODE_RANK, s_all_nodes );
  for ( stk::mesh::BucketVector::const_iterator ib = node_buckets.begin() ;
        ib != node_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib ;
    const stk::mesh::Bucket::size_type length  = b.size();
    const double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, b);
    double * bcTurbKineticEnergy = stk::mesh::field_data(*bcTurbKineticEnergy_, b);
    double * bcAssembledTurbKineticEnergy = stk::mesh::field_data(*bcAssembledTurbKineticEnergy_, b);
    double * tke = stk::mesh::field_data(tkeNp1, b);
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      const double tkeBnd = bcAssembledTurbKineticEnergy[k]/assembledWallArea[k];
      bcAssembledTurbKineticEnergy[k] = tkeBnd;
      bcTurbKineticEnergy[k] = tkeBnd;
      // make sure that the next matrix assembly uses the proper tke value
      tke[k] = tkeBnd;
    }
  }
}

} // namespace nalu
} // namespace Sierra
