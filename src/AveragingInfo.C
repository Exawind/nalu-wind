// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <AveragingInfo.h>
#include <NaluParsing.h>

// basic c++
#include <stdexcept>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// AveragingInfo - holder for averaging information held at TurbAvePP
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AveragingInfo::AveragingInfo()
  : computeReynoldsStress_(false),
    computeTke_(false),
    computeFavreStress_(false),
    computeFavreTke_(false),
    computeVorticity_(false),
    computeQcriterion_(false),
    computeLambdaCI_(false),
    computeMeanResolvedKe_(false)
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
AveragingInfo::~AveragingInfo()
{
  // nothing to do
}

} // namespace nalu
} // namespace sierra
