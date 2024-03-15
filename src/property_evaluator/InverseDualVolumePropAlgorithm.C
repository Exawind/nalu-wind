// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Algorithm.h>
#include <property_evaluator/InverseDualVolumePropAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace nalu {

InverseDualVolumePropAlgorithm::InverseDualVolumePropAlgorithm(
  Realm& realm, stk::mesh::Part* part, stk::mesh::FieldBase* prop)
  : Algorithm(realm, part), prop_(prop), dualNodalVolume_(NULL)
{
  // extract dual volume
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  dualNodalVolume_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, "dual_nodal_volume");
}

InverseDualVolumePropAlgorithm::~InverseDualVolumePropAlgorithm() {}

void
InverseDualVolumePropAlgorithm::execute()
{

  // make sure that partVec_ is size one
  ThrowAssert(partVec_.size() == 1);

  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double* prop = (double*)stk::mesh::field_data(*prop_, b);
    const double* dualNodalVolume =
      (double*)stk::mesh::field_data(*dualNodalVolume_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      prop[k] = 1.0 / dualNodalVolume[k];
    }
  }
}

} // namespace nalu
} // namespace sierra
