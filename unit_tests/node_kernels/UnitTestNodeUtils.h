// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTNODEUTILS_H
#define UNITTESTNODEUTILS_H

#include "wind_energy/ABLForcingAlgorithm.h"
#include "Realm.h"

namespace unit_test_utils {

class TestABLForcingAlg : public sierra::nalu::ABLForcingAlgorithm
{
public:
  TestABLForcingAlg(sierra::nalu::Realm& realm) : ABLForcingAlgorithm(realm)
  {
    USource_ = {{10.0}, {10.0}, {10.0}};
    TSource_.clear();
    TSource_.push_back(300.0);

    std::vector<double> heights(1, 90.0);

    USrcInterp_.reset(
      new sierra::nalu::ABLVectorInterpolator(heights, USource_));
    TSrcInterp_.reset(
      new sierra::nalu::ABLScalarInterpolator(heights, TSource_));
  }
};

} // namespace unit_test_utils

#endif /* UNITTESTNODEUTILS_H */
