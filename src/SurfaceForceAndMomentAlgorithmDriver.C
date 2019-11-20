/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <SurfaceForceAndMomentAlgorithmDriver.h>
#include <Algorithm.h>
#include <AlgorithmDriver.h>
#include <FieldFunctions.h>
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

namespace sierra{
namespace nalu{

class Realm;

//==========================================================================
// Class Definition
//==========================================================================
// SurfaceForceAndMomentAlgorithmDriver - Drives fluids post processing algs
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
SurfaceForceAndMomentAlgorithmDriver::SurfaceForceAndMomentAlgorithmDriver(
  Realm &realm)
  : AlgorithmDriver(realm)
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
SurfaceForceAndMomentAlgorithmDriver::~SurfaceForceAndMomentAlgorithmDriver()
{
  std::vector<Algorithm *>::iterator iter, iter_end;
  iter_end = algVec_.end();
  for(iter = algVec_.begin(); iter != iter_end; ++iter)
    delete *iter;
}

//--------------------------------------------------------------------------
//-------- zero_fields -------------------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentAlgorithmDriver::zero_fields()
{

  // common
  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  
  // extract the fields
  VectorFieldType *pressureForce = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "pressure_force");
  VectorFieldType *viscousForce = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "viscous_force");
  ScalarFieldType *tauWall = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tau_wall");
  ScalarFieldType *yplus = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "yplus");
  // one of these might be null
  ScalarFieldType *assembledArea = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment");
  ScalarFieldType *assembledAreaWF = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment_wf");

  // zero fields
  field_fill( meta_data, bulk_data, 0.0, *pressureForce, realm_.get_activate_aura());
  field_fill( meta_data, bulk_data, 0.0, *viscousForce, realm_.get_activate_aura());
  field_fill( meta_data, bulk_data, 0.0, *tauWall, realm_.get_activate_aura());
  field_fill( meta_data, bulk_data, 0.0, *yplus, realm_.get_activate_aura());
  if ( NULL != assembledArea ) 
    field_fill( meta_data, bulk_data, 0.0, *assembledArea, realm_.get_activate_aura());
  if ( NULL != assembledAreaWF ) 
    field_fill( meta_data, bulk_data, 0.0, *assembledAreaWF, realm_.get_activate_aura());

}

//--------------------------------------------------------------------------
//-------- parralel_assemble_fields ----------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentAlgorithmDriver::parallel_assemble_fields()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  const size_t nDim = meta_data.spatial_dimension();

  // extract the fields
  VectorFieldType *pressureForce = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "pressure_force");
  VectorFieldType *viscousForce = meta_data.get_field<VectorFieldType>(stk::topology::NODE_RANK, "viscous_force");
  ScalarFieldType *tauWall = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "tau_wall");
  ScalarFieldType *yplus = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "yplus");

  stk::mesh::parallel_sum(bulk_data, {pressureForce, viscousForce, tauWall, yplus});

  // periodic assemble
  if ( realm_.hasPeriodic_) {
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    realm_.periodic_field_update(pressureForce, nDim, bypassFieldCheck);
    realm_.periodic_field_update(viscousForce, nDim, bypassFieldCheck);
    realm_.periodic_field_update(tauWall, 1, bypassFieldCheck);
    realm_.periodic_field_update(yplus, 1, bypassFieldCheck);
  }

}

//--------------------------------------------------------------------------
//-------- parallel_assemble_area ------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentAlgorithmDriver::parallel_assemble_area()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  // extract the fields; one of these might be null
  ScalarFieldType *assembledArea = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment");
  ScalarFieldType *assembledAreaWF = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_area_force_moment_wf");

  // parallel assemble
  std::vector<const stk::mesh::FieldBase*> fields;
  if ( NULL != assembledArea )
    fields.push_back(assembledArea);
  if ( NULL != assembledAreaWF )
    fields.push_back(assembledAreaWF);
  const std::vector<const stk::mesh::FieldBase*>& const_fields = fields;
  stk::mesh::parallel_sum(bulk_data, const_fields);

  // periodic assemble
  if ( realm_.hasPeriodic_) {
    const bool bypassFieldCheck = false; // fields are not defined at all slave/master node pairs
    if ( NULL != assembledArea )
      realm_.periodic_field_update(assembledArea, 1, bypassFieldCheck);
    if ( NULL != assembledAreaWF )
      realm_.periodic_field_update(assembledAreaWF, 1, bypassFieldCheck);
  }

}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
SurfaceForceAndMomentAlgorithmDriver::execute()
{

  // zero fields
  zero_fields();

  // pre-work; basically, allow algs to load up nodal area(s)
  for ( size_t k = 0; k < algVec_.size(); ++k )
    algVec_[k]->pre_work();

  // assemble area
  parallel_assemble_area();

  // execute
  for ( size_t k = 0; k < algVec_.size(); ++k )
    algVec_[k]->execute();
  
  // parallel assembly
  parallel_assemble_fields();

}


} // namespace nalu
} // namespace Sierra
