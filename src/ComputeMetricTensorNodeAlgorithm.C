// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


// nalu
#include <Algorithm.h>
#include <ComputeMetricTensorNodeAlgorithm.h>

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
// ComputeMetricTensorNodeAlgorithm - Metric Tensor
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMetricTensorNodeAlgorithm::ComputeMetricTensorNodeAlgorithm(
    Realm &realm, stk::mesh::Part *part)
    : Algorithm(realm, part) {
  // save off data
  stk::mesh::MetaData &meta_data = realm_.meta_data();
  coordinates_ = meta_data.get_field<VectorFieldType>(
      stk::topology::NODE_RANK, realm_.get_coordinates_name());
  nodalMij_ = meta_data.get_field<GenericFieldType>(stk::topology::NODE_RANK,
                                               "metric_tensor");
  dualNodalVolume_ = meta_data.get_field<ScalarFieldType>(stk::topology::NODE_RANK, 
                                               "dual_nodal_volume");
}

ComputeMetricTensorNodeAlgorithm::~ComputeMetricTensorNodeAlgorithm()
{
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void ComputeMetricTensorNodeAlgorithm::execute() {

  stk::mesh::BulkData & bulk_data = realm_.bulk_data();
  stk::mesh::MetaData &meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  stk::mesh::Selector selector =
      meta_data.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const &node_buckets =
      realm_.get_buckets(stk::topology::NODE_RANK, selector);
  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket &b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      // get Mij field_data
      double *Mij = stk::mesh::field_data(*nodalMij_, b[k]);

      // initialize to 0
      for (int i = 0; i < nDim; ++i)
        for (int j = 0; j < nDim; ++j)
          Mij[i * nDim + j] = 0.0;
    }
  }

  stk::mesh::BucketVector const &elem_buckets =
      realm_.get_buckets(stk::topology::ELEMENT_RANK, selector);
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
    const int numScvIp = meSCV->num_integration_points();

    // resize std::vectors based on element type
    ws_coordinates.resize(nDim * nodesPerElement);
    ws_dndx.resize(nDim * numScvIp * nodesPerElement);
    ws_deriv.resize(nDim * numScvIp * nodesPerElement);
    ws_det_j.resize(numScvIp);
    ws_scv_volume.resize(numScvIp);
    ws_Mij.resize(numScvIp * nDim * nDim);

    // pointers to vectors
    double *p_coords = &ws_coordinates[0];
    double *p_Mij = &ws_Mij[0];

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {

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
      meSCV->determinant(1, &p_coords[0], &ws_scv_volume[0], &scv_error);
      const int *ipNodeMap = meSCV->ipNodeMap();

      // since we only want a single elemental value, average over all
      // integration points
      for (int ip = 0; ip < numScvIp; ++ip) {
 
        // nearest node to ip
        stk::mesh::Entity nearestNode = node_rels[ipNodeMap[ip]];

        const double *dualNodalVolume = stk::mesh::field_data(*dualNodalVolume_, nearestNode);
        double * nodalMij = stk::mesh::field_data(*nodalMij_, nearestNode); 

        const double scV = ws_scv_volume[ip];
        for (int i = 0; i < nDim; ++i)
          for (int j = 0; j < nDim; ++j)
            nodalMij[i * nDim + j] +=
                p_Mij[ip * nDim * nDim + i * nDim + j] * scV/dualNodalVolume[0];
      }
    }
  }

  stk::mesh::parallel_sum(bulk_data, {nodalMij_});

  if ( realm_.hasPeriodic_) {
    const unsigned nDim = meta_data.spatial_dimension();
    realm_.periodic_field_update(nodalMij_, nDim*nDim);
  }
}

} // namespace nalu
} // namespace sierra
