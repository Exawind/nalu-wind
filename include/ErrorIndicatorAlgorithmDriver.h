// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ErrorIndicatorAlgorithmDriver_h
#define ErrorIndicatorAlgorithmDriver_h

#include <AlgorithmDriver.h>
#include <FieldTypeDef.h>

#if defined (NALU_USES_PERCEPT)
#include <percept/FieldTypes.hpp>
#endif

namespace sierra{
namespace nalu{

class Realm;

class ErrorIndicatorAlgorithmDriver : public AlgorithmDriver
{
public:

  ErrorIndicatorAlgorithmDriver(
    Realm &realm);
  ~ErrorIndicatorAlgorithmDriver();

  void pre_work();
  void post_work();

public:

#if defined (NALU_USES_PERCEPT)
  GenericFieldType *errorIndicator_;
  percept::RefineFieldType *refineField_;
  percept::RefineFieldType *refineFieldOrig_;
  percept::RefineLevelType *refineLevelField_;
#else
  GenericFieldType *errorIndicator_;
  ScalarFieldType *refineField_;
  ScalarFieldType *refineFieldOrig_;
  ScalarFieldType *refineLevelField_;
#endif

  double maxErrorIndicator_;
};

} // namespace nalu
} // namespace Sierra

#endif

