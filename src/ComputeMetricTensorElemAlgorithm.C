/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

// nalu
#include <Algorithm.h>
#include <ComputeMetricTensorElemAlgorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <master_element/MasterElementFactory.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ComputeMetricTensorElemAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMetricTensorElemAlgorithm::ComputeMetricTensorElemAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part) {
  // save off data
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, realm_.get_coordinates_name());
  Mij_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK,
                                               "metric_tensor");
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeMetricTensorElemAlgorithm::execute() {

  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // fill in elemental values
  stk::mesh::Selector s_locally_owned_union =
      meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const &elem_buckets =
      realm_.get_buckets(stk::topology::ELEMENT_RANK, s_locally_owned_union);
  for (stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
       ib != elem_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    // extract master element
    MasterElement *meSCV =
        sierra::nalu::MasterElementRepo::get_volume_master_element(
            b.topology());

    // extract master element specifics
    const int nodesPerElement = meSCV->nodesPerElement_;
    const int numScvIp = meSCV->numIntPoints_;

    // resize std::vectors based on element type
    ws_coordinates.resize(nDim * nodesPerElement);
    ws_dndx.resize(nDim * numScvIp * nodesPerElement);
    ws_deriv.resize(nDim * numScvIp * nodesPerElement);
    ws_det_j.resize(numScvIp);
    ws_Mij.resize(numScvIp * nDim * nDim);

    // pointers to vectors
    double *p_coords = &ws_coordinates[0];
    double *p_Mij = &ws_Mij[0];

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

      // get Mij field_data
      double *Mij = stk::mesh::field_data(*Mij_, b[k]);

      // initialize to 0
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          Mij[i * nDim + j] = 0.0;

      //===============================================
      // gather nodal data; this is how we do it now..
      //===============================================
      stk::mesh::Entity const *node_rels = b.begin_nodes(k);
      int num_nodes = b.num_nodes(k);

      // sanity check on num nodes
      ThrowAssert(num_nodes == nodesPerElement);

      // loop over nodes to create coords vector
      for (int ni = 0; ni < num_nodes; ++ni) {
        stk::mesh::Entity node = node_rels[ni];

        // pointers to real data
        const double *coords = stk::mesh::field_data(*coordinates_, node);

        // gather coords vector
        const int niNdim = ni * nDim;
        for (int j = 0; j < nDim; ++j)
          p_coords[niNdim + j] = coords[j];
      }

      // compute geometry and get Mij from master element function at
      // integration points
      double scv_error = 0.0;
      meSCV->grad_op(1, &p_coords[0], &ws_dndx[0], &ws_deriv[0], &ws_det_j[0],
                     &scv_error);
      meSCV->Mij(&p_coords[0], &p_Mij[0], &ws_deriv[0]);

      // since we only want a single elemental value, average over all
      // integration points
      for (int ip = 0; ip < numScvIp; ++ip)
        for (int i = 0; i < nDim; ++i)
          for (int j = 0; j < nDim; ++j)
            Mij[i * nDim + j] +=
                p_Mij[ip * nDim * nDim + i * nDim + j] / numScvIp;
    }
  }
}

} // namespace nalu
} // namespace sierra
