/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <AssembleNodalGradPAlgorithmDriver.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>


// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleNodalGradPAlgorithmDriver - Drives nodal grad algorithms
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleNodalGradPAlgorithmDriver::AssembleNodalGradPAlgorithmDriver(Realm &realm)
  : AlgorithmDriver(realm)
{
}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradPAlgorithmDriver::pre_work()
{
  ThrowRequire(realm_.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx") != nullptr);
  VectorFieldType& dpdxField = *realm_.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");
  stk::mesh::field_fill(0.0, dpdxField);
}

//--------------------------------------------------------------------------
//-------- post_work -------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleNodalGradPAlgorithmDriver::post_work()
{

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  ThrowRequire(realm_.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx") != nullptr);
  VectorFieldType& dpdxField = *realm_.meta_data().get_field<VectorFieldType>(stk::topology::NODE_RANK, "dpdx");

  // extract fields
  stk::mesh::parallel_sum(bulk_data, {&dpdxField});

  if ( realm_.hasPeriodic_) {
    const unsigned nDim = meta_data.spatial_dimension();
    realm_.periodic_field_update(&dpdxField, nDim);
  }

  if ( realm_.hasOverset_ ) {
    // this is a tensor
    const unsigned nDim = meta_data.spatial_dimension();
    realm_.overset_orphan_node_field_update(&dpdxField, 1, nDim);
  }

}

} // namespace nalu
} // namespace Sierra
