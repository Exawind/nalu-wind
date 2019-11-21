// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/LinearRampMeshDisplacementAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

LinearRampMeshDisplacementAuxFunction::LinearRampMeshDisplacementAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double> theParams) :
  AuxFunction(beginPos, endPos),
  ramp_(0)
{
  size_t paramSize = theParams.size();
  ramp_.resize(paramSize);
  for ( size_t k = 0; k < paramSize; ++k)
    ramp_[k] = theParams[k];
}


void
LinearRampMeshDisplacementAuxFunction::do_evaluate(
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
    for ( unsigned k = 0; k < fieldSize; ++k )
      fieldPtr[k] = ramp_[k]*time;
    fieldPtr += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
