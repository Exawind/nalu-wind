// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Algorithm.h>
#include <property_evaluator/GenericPropAlgorithm.h>
#include <FieldTypeDef.h>
#include <property_evaluator/PropertyEvaluator.h>
#include <property_evaluator/ConstantPropertyEvaluator.h>
#include <Realm.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

namespace sierra {
namespace nalu {

GenericPropAlgorithm::GenericPropAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  stk::mesh::FieldBase* prop,
  PropertyEvaluator* propEvaluator)
  : Algorithm(realm, part), prop_(prop), propEvaluator_(propEvaluator)
{
  // does nothing
}

void
GenericPropAlgorithm::execute()
{
  const bool is_constant_prop =
    dynamic_cast<ConstantPropertyEvaluator*>(propEvaluator_) != nullptr;
  if (is_constant_prop) {
    const double val =
      dynamic_cast<ConstantPropertyEvaluator*>(propEvaluator_)->value_;
    auto prop_ngp = stk::mesh::get_updated_ngp_field<double>(*prop_);
    prop_ngp.set_all(realm_.ngp_mesh(), val);
    prop_ngp.modify_on_device();
    return;
  }

  // make sure that partVec_ is size one
  STK_ThrowAssert(partVec_.size() == 1);

  // empty independet variable list; hence "Generic"
  std::vector<double> indVarList(1, 0.0);

  stk::mesh::Selector selector = stk::mesh::selectUnion(partVec_);

  stk::mesh::BucketVector const& node_buckets =
    realm_.get_buckets(stk::topology::NODE_RANK, selector);

  for (stk::mesh::BucketVector::const_iterator ib = node_buckets.begin();
       ib != node_buckets.end(); ++ib) {
    stk::mesh::Bucket& b = **ib;
    const stk::mesh::Bucket::size_type length = b.size();

    double* prop = (double*)stk::mesh::field_data(*prop_, b);

    for (stk::mesh::Bucket::size_type k = 0; k < length; ++k) {
      prop[k] = propEvaluator_->execute(&indVarList[0], b[k]);
    }
  }
}

} // namespace nalu
} // namespace sierra
