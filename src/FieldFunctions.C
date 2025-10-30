// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <FieldFunctions.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Field.hpp>

#include <algorithm>

namespace sierra {
namespace nalu {

void
field_axpby(
  const stk::mesh::MetaData& metaData,
  const stk::mesh::BulkData& bulkData,
  const double alpha,
  const stk::mesh::FieldBase& xField,
  const double beta,
  const stk::mesh::FieldBase& yField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector =
    auraIsActive
      ? metaData.universal_part() & stk::mesh::selectField(xField) &
          stk::mesh::selectField(yField)
      : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
          stk::mesh::selectField(xField) & stk::mesh::selectField(yField);

  stk::mesh::BucketVector const& buckets =
    bulkData.get_buckets(entityRankValue, selector);

  for (size_t i = 0; i < buckets.size(); ++i) {
    stk::mesh::Bucket& b = *buckets[i];
    const stk::mesh::Bucket::size_type length = b.size();
    const size_t fieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    STK_ThrowAssert(
      fieldSize == field_bytes_per_entity(yField, b) / sizeof(double));
    const unsigned kmax = length * fieldSize;
    const double* x = (double*)stk::mesh::field_data(xField, b);
    double* y = (double*)stk::mesh::field_data(yField, b);
    for (unsigned k = 0; k < kmax; ++k) {
      y[k] = alpha * x[k] + beta * y[k];
    }
  }
}

void
field_fill(
  const stk::mesh::MetaData& metaData,
  const stk::mesh::BulkData& bulkData,
  const double alpha,
  const stk::mesh::FieldBase& xField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector =
    auraIsActive
      ? metaData.universal_part() & stk::mesh::selectField(xField)
      : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
          stk::mesh::selectField(xField);

  stk::mesh::BucketVector const& buckets =
    bulkData.get_buckets(entityRankValue, selector);

  for (size_t i = 0; i < buckets.size(); ++i) {
    stk::mesh::Bucket& b = *buckets[i];
    const stk::mesh::Bucket::size_type length = b.size();
    const unsigned fieldSize =
      field_bytes_per_entity(xField, b) / sizeof(double);
    const unsigned kmax = length * fieldSize;
    double* x = (double*)stk::mesh::field_data(xField, b);
    std::fill(x, x + kmax, alpha);
  }
}

void
field_scale(
  const stk::mesh::MetaData& metaData,
  const stk::mesh::BulkData& bulkData,
  const double alpha,
  const stk::mesh::FieldBase& xField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector =
    auraIsActive
      ? metaData.universal_part() & stk::mesh::selectField(xField)
      : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
          stk::mesh::selectField(xField);

  stk::mesh::BucketVector const& buckets =
    bulkData.get_buckets(entityRankValue, selector);

  for (size_t i = 0; i < buckets.size(); ++i) {
    stk::mesh::Bucket& b = *buckets[i];
    const stk::mesh::Bucket::size_type length = b.size();
    const unsigned fieldSize =
      field_bytes_per_entity(xField, b) / sizeof(double);
    const unsigned kmax = length * fieldSize;
    double* x = (double*)stk::mesh::field_data(xField, b);
    for (unsigned k = 0; k < kmax; ++k) {
      x[k] = alpha * x[k];
    }
  }
}

void
field_copy(
  const stk::mesh::MetaData& metaData,
  const stk::mesh::BulkData& bulkData,
  const stk::mesh::FieldBase& xField,
  const stk::mesh::FieldBase& yField,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector =
    auraIsActive
      ? metaData.universal_part() & stk::mesh::selectField(xField) &
          stk::mesh::selectField(yField)
      : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
          stk::mesh::selectField(xField) & stk::mesh::selectField(yField);

  stk::mesh::BucketVector const& buckets =
    bulkData.get_buckets(entityRankValue, selector);

  for (size_t i = 0; i < buckets.size(); ++i) {
    stk::mesh::Bucket& b = *buckets[i];
    const stk::mesh::Bucket::size_type length = b.size();
    const size_t fieldSize = field_bytes_per_entity(xField, b) / sizeof(double);
    STK_ThrowAssert(
      fieldSize == field_bytes_per_entity(yField, b) / sizeof(double));
    const unsigned kmax = length * fieldSize;
    const double* x = (double*)stk::mesh::field_data(xField, b);
    double* y = (double*)stk::mesh::field_data(yField, b);
    for (unsigned k = 0; k < kmax; ++k) {
      y[k] = x[k];
    }
  }
}

void
field_index_copy(
  const stk::mesh::MetaData& metaData,
  const stk::mesh::BulkData& bulkData,
  const stk::mesh::FieldBase& xField,
  const int xFieldIndex,
  const stk::mesh::FieldBase& yField,
  const int yFieldIndex,
  const bool auraIsActive,
  const stk::topology::rank_t entityRankValue)
{
  // decide on selector
  const stk::mesh::Selector selector =
    auraIsActive
      ? metaData.universal_part() & stk::mesh::selectField(xField) &
          stk::mesh::selectField(yField)
      : (metaData.locally_owned_part() | metaData.globally_shared_part()) &
          stk::mesh::selectField(xField) & stk::mesh::selectField(yField);

  stk::mesh::BucketVector const& buckets =
    bulkData.get_buckets(entityRankValue, selector);

  for (size_t i = 0; i < buckets.size(); ++i) {
    stk::mesh::Bucket& b = *buckets[i];
    const stk::mesh::Bucket::size_type length = b.size();
    const size_t xFieldSize =
      field_bytes_per_entity(xField, b) / sizeof(double);
    const size_t yFieldSize =
      field_bytes_per_entity(yField, b) / sizeof(double);
    const double* x = (double*)stk::mesh::field_data(xField, b);
    double* y = (double*)stk::mesh::field_data(yField, b);
    for (unsigned k = 0; k < length; ++k) {
      y[k * yFieldSize + yFieldIndex] = x[k * xFieldSize + xFieldIndex];
    }
  }
}

} // namespace nalu
} // namespace sierra
