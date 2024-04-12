// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Algorithm.h>
#include <property_evaluator/LinearPropAlgorithm.h>
#include <FieldTypeDef.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace nalu {

LinearPropAlgorithm::LinearPropAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  stk::mesh::FieldBase* prop,
  stk::mesh::FieldBase* indVar,
  const double primary,
  const double secondary)
  : Algorithm(realm, part),
    prop_(prop),
    indVar_(indVar),
    primary_(primary),
    secondary_(secondary)
{
  // does nothing
}

void
LinearPropAlgorithm::execute()
{

  // make sure that partVec_ is size one
  STK_ThrowAssert(partVec_.size() == 1);

  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double* prop = (double*)stk::mesh::field_data(*prop_, b);
    const double* indVar = (double*)stk::mesh::field_data(*indVar_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      const double z = indVar[k];
      const double om_z = 1.0 - z;
      prop[k] = z * primary_ + om_z * secondary_;
    }
  }
}

} // namespace nalu
} // namespace sierra
