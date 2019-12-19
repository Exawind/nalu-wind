// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <ErrorIndicatorAlgorithmDriver.h>
#include <Algorithm.h>
#include <AlgorithmDriver.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <SolutionOptions.h>

#if defined (NALU_USES_PERCEPT)
#include <adapt/markers/MarkerUsingErrIndFraction.hpp>
#include <adapt/markers/MarkerPhysicallyBased.hpp>
#include <percept/FieldTypes.hpp>
#endif

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/Comm.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

#include <algorithm>
#include <functional>

namespace sierra{
namespace nalu{

class Realm;

//==========================================================================
// Class Definition
//==========================================================================
// ErrorIndicatorAlgorithmDriver - Drives nodal grad algorithms
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ErrorIndicatorAlgorithmDriver::ErrorIndicatorAlgorithmDriver(
  Realm &realm)
  : AlgorithmDriver(realm),
    errorIndicator_(NULL), refineField_(NULL), refineFieldOrig_(NULL), refineLevelField_(NULL), maxErrorIndicator_(0.0)
{
  // save off fields
#if defined (NALU_USES_PERCEPT)
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  errorIndicator_ = meta_data.get_field<GenericFieldType>(stk::topology::ELEMENT_RANK, "error_indicator");
  if (realm.solutionOptions_->useMarker_)
    {
      refineField_ = meta_data.get_field<percept::RefineFieldType>(stk::topology::ELEMENT_RANK, "refine_field");
      refineFieldOrig_ = meta_data.get_field<percept::RefineFieldType>(stk::topology::ELEMENT_RANK, "refine_field_orig");
      refineLevelField_ = meta_data.get_field<percept::RefineLevelType>(stk::topology::ELEMENT_RANK, "refine_level");
    }
#endif
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ErrorIndicatorAlgorithmDriver::~ErrorIndicatorAlgorithmDriver()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- pre_work --------------------------------------------------------
//--------------------------------------------------------------------------
void
ErrorIndicatorAlgorithmDriver::pre_work()
{
  // nothing to do
}

//--------------------------------------------------------------------------
//-------- post_work -------------------------------------------------------
//--------------------------------------------------------------------------

void
ErrorIndicatorAlgorithmDriver::post_work()
{

#if defined (NALU_USES_PERCEPT)

  // Marker
  // different criteria for refinement - we just pick #1 for now
  /**
   * 1. refine top R%, unrefine bottom U% of error indicator
   * 2. refine top NR% of elements, unrefine bottom NU% of elements
   * ... etc...
   */

  percept::MarkerInfo markerInfo;
  markerInfo.errorIndicator_ = this->errorIndicator_;
  markerInfo.refineField_ = this->refineField_;
  markerInfo.refineFieldOrig_ = this->refineFieldOrig_;
  markerInfo.refineLevelField_ = this->refineLevelField_;
  markerInfo.useMarker_ = realm_.solutionOptions_->useMarker_;
  markerInfo.numInitialElements_ = realm_.numInitialElements_;
  markerInfo.maxRefinementLevel_ = realm_.solutionOptions_->maxRefinementLevel_;
  markerInfo.maxRefinementNumberOfElementsFraction_ = realm_.solutionOptions_->maxRefinementNumberOfElementsFraction_;
  markerInfo.debug_ = realm_.debug();

  if (realm_.solutionOptions_->errorIndicatorType_ == EIT_SIMPLE_VORTICITY_DX) {
    markerInfo.physicalErrIndCriterion_ = realm_.solutionOptions_->physicalErrIndCriterion_;
    markerInfo.physicalErrIndUnrefCriterionMultipler_ = realm_.solutionOptions_->physicalErrIndUnrefCriterionMultipler_;

    percept::MarkerPhysicallyBased marker(realm_.bulk_data(), markerInfo);
    marker.mark();
  }
  else {
    markerInfo.refineFraction_ = realm_.solutionOptions_->refineFraction_;
    markerInfo.unrefineFraction_ = realm_.solutionOptions_->unrefineFraction_;

    percept::MarkerUsingErrIndFraction marker(realm_.bulk_data(), markerInfo);
    marker.mark();
  }
# endif
}

} // namespace nalu
} // namespace Sierra

