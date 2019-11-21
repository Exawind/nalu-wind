// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/SinMeshDisplacementAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

SinMeshDisplacementAuxFunction::SinMeshDisplacementAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double> theParams) :
  AuxFunction(beginPos, endPos),
  pi_(acos(-1.0)),
  maxDisplacement_(0)
{
  maxDisplacement_ = theParams[0];
}


void
SinMeshDisplacementAuxFunction::do_evaluate(
  const double */*coords*/,
  const double time,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {
    fieldPtr[0] = 0.0;
    fieldPtr[1] = sin(pi_*time)*maxDisplacement_;
    fieldPtr += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
