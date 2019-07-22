/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTNODEUTILS_H
#define UNITTESTNODEUTILS_H

#include "wind_energy/ABLForcingAlgorithm.h"
#include "Realm.h"

namespace unit_test_utils {

class TestABLForcingAlg : public sierra::nalu::ABLForcingAlgorithm
{
public:
  TestABLForcingAlg(
    sierra::nalu::Realm& realm
  ) : ABLForcingAlgorithm(realm)
  {
    USource_ = {{10.0}, {10.0}, {10.0}};
    TSource_ = { 300.0 };

    std::vector<double> heights(1, 90.0);

    USrcInterp_.reset(new sierra::nalu::ABLVectorInterpolator(heights, USource_));
    TSrcInterp_.reset(new sierra::nalu::ABLScalarInterpolator(heights, TSource_));
  }
};

}

#endif /* UNITTESTNODEUTILS_H */
