// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Algorithm.h>
#include <property_evaluator/TemperaturePropAlgorithm.h>
#include <FieldTypeDef.h>
#include <property_evaluator/PropertyEvaluator.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace nalu {

TemperaturePropAlgorithm::TemperaturePropAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  stk::mesh::FieldBase* prop,
  PropertyEvaluator* propEvaluator,
  std::string tempName)
  : Algorithm(realm, part),
    prop_(prop),
    propEvaluator_(propEvaluator),
    temperature_(NULL)
{
  // extract temperature field
  stk::mesh::MetaData& meta_data = realm_.meta_data();
  temperature_ =
    meta_data.get_field<double>(stk::topology::NODE_RANK, tempName);
  if (NULL == temperature_) {
    throw std::runtime_error("Realm::setup_property: TemperaturePropAlgorithm "
                             "requires temperature/bc:");
  }
}

void
TemperaturePropAlgorithm::execute()
{

  // make sure that partVec_ is size one
  ThrowAssert(partVec_.size() == 1);

  std::vector<double> indVarList(1);

  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);

  prop_->sync_to_host();
  temperature_->sync_to_host();

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double* prop = (double*)stk::mesh::field_data(*prop_, b);
    const double* temperature =
      (double*)stk::mesh::field_data(*temperature_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      indVarList[0] = temperature[k];
      prop[k] = propEvaluator_->execute(&indVarList[0], b[k]);
    }
  }
}

} // namespace nalu
} // namespace sierra
