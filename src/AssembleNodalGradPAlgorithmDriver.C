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

class Realm;


void subtract_normal_component(int dim, const double* normal, double* grad_p)
{
  double grad_p_dot_n = 0;
  for (int j = 0; j < dim; ++j) {
    grad_p_dot_n += normal[j] * grad_p[j];
  }

  for (int j = 0; j < dim; ++j) {
    grad_p[j] -= grad_p_dot_n * normal[j];
  }
}

void subtract_normal_pressure_gradient(const stk::mesh::BulkData& bulk, VectorFieldType& dqdxField)
{
  const auto& meta = bulk.mesh_meta_data();
  ThrowRequireMsg(meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "average_open_normal") != nullptr,
    "average_open_normal field required");
  VectorFieldType& meanNormalField = *meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "average_open_normal");

  const int nDim = meta.spatial_dimension();

  const stk::mesh::Selector& owned_or_shared_open = (meta.locally_owned_part() | meta.globally_shared_part())
      & stk::mesh::selectField(meanNormalField);
  const stk::mesh::BucketVector& node_buckets = bulk.get_buckets(stk::topology::NODE_RANK, owned_or_shared_open);
  for (const auto* ib : node_buckets) {
    const auto& b = *ib;
    const size_t length = b.size();
    for (size_t k = 0u; k < length; ++k) {
      const stk::mesh::Entity node = b[k];
      const double* meanNormal = stk::mesh::field_data(meanNormalField, node);
      double* dqdx = stk::mesh::field_data(dqdxField, node);
      subtract_normal_component(nDim, meanNormal, dqdx);
    }
  }
}

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

  if (realm_.solutionOptions_->explicitlyZeroOpenPressureGradient_) {
    subtract_normal_pressure_gradient(bulk_data, dpdxField);
  }

}

} // namespace nalu
} // namespace Sierra
