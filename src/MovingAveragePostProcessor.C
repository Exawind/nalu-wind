// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

// nalu
#include <MovingAveragePostProcessor.h>
#include <Algorithm.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <TimeIntegrator.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

#include <limits>

namespace sierra {
namespace nalu {

//--------------------------------------------------------------------------
ExponentialMovingAverager::ExponentialMovingAverager(
  double timeScale, bool isInit, double alpha)
  : timeScale_(timeScale), isInit_(isInit), alpha_(alpha)
{
}
//--------------------------------------------------------------------------
void
ExponentialMovingAverager::compute_and_set_alpha(double delta_t)
{
  STK_ThrowAssert(delta_t > 1.0e6 * std::numeric_limits<double>::min());
  alpha_ = (isInit_) ? 1.0 : std::min(1.0, delta_t / timeScale_);
}
//--------------------------------------------------------------------------
double
ExponentialMovingAverager::compute_updated_average(double oldAvg, double newVal)
{
  return (alpha_ * newVal + (1 - alpha_) * oldAvg);
}
//---------------------------------------------------------------------
void
ExponentialMovingAverager::init_state(bool init)
{
  isInit_ = init;
}
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
MovingAveragePostProcessor::MovingAveragePostProcessor(
  stk::mesh::BulkData& bulk, TimeIntegrator& timeIntegrator, bool isRestarted)
  : bulk_(bulk), timeIntegrator_(timeIntegrator), isRestarted_(isRestarted)
{
}
//--------------------------------------------------------------------------
void
MovingAveragePostProcessor::add_fields(std::vector<std::string> fieldNames)
{
  for (const auto& fieldName : fieldNames) {
    auto& meta = bulk_.mesh_meta_data();
    auto* field = meta.get_field(stk::topology::NODE_RANK, fieldName);
    STK_ThrowRequireMsg(
      field != nullptr,
      "Requested field `" + fieldName + "' not available for averaging");
    STK_ThrowRequireMsg(
      field->type_is<double>(), "Only double precision-typed fields allowed");

    stk::mesh::FieldBase* avgField = meta.get_field(
      stk::topology::NODE_RANK, filtered_field_name(field->name()));
    STK_ThrowRequireMsg(
      avgField != nullptr,
      filtered_field_name(field->name()) + " field not registered");
    fieldMap_.insert({field, avgField});
  }
}
//--------------------------------------------------------------------------
void
MovingAveragePostProcessor::set_time_scale(
  std::string fieldName, double timeScale)
{
  averagers_[fieldName] = ExponentialMovingAverager(timeScale, !isRestarted_);
}
//--------------------------------------------------------------------------
void
MovingAveragePostProcessor::set_time_scale(double timeScale)
{
  for (const auto& fieldPair : fieldMap_) {
    averagers_[fieldPair.first->name()] =
      ExponentialMovingAverager(timeScale, !isRestarted_);
  }
}
//--------------------------------------------------------------------------
void
MovingAveragePostProcessor::execute()
{
  for (auto& fieldPair : fieldMap_) {
    const auto& field = *fieldPair.first;
    auto& avgField = *fieldPair.second;
    auto& averager = averagers_.at(field.name());
    averager.compute_and_set_alpha(timeIntegrator_.get_time_step());

    // average wherever the field is defined
    const auto& node_buckets = bulk_.get_buckets(
      stk::topology::NODE_RANK, stk::mesh::selectField(field));
    for (const auto* ib : node_buckets) {
      const auto& b = *ib;
      const double* fieldVals =
        static_cast<const double*>(stk::mesh::field_data(field, b));
      double* avgFieldVals =
        static_cast<double*>(stk::mesh::field_data(avgField, b));
      const unsigned fieldSize = stk::mesh::field_scalars_per_entity(field, b);
      STK_ThrowAssert(
        fieldSize == stk::mesh::field_scalars_per_entity(avgField, b));
      for (size_t k = 0u; k < b.size(); ++k) {
        const size_t offset = k * fieldSize;
        for (unsigned d = 0; d < fieldSize; ++d) {
          avgFieldVals[d + offset] = averager.compute_updated_average(
            avgFieldVals[d + offset], fieldVals[d + offset]);
        }
      }
    }
    averager.init_state(false);
  }
}

} // namespace nalu
} // namespace sierra
