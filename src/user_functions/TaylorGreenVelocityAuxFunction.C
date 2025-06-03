// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/TaylorGreenVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

TaylorGreenVelocityAuxFunction::TaylorGreenVelocityAuxFunction(
  const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos), uNot_(1.0), L_(1.0)
{
  // must be 3D
  if (3 != endPos)
    throw std::runtime_error(
      "TaylorGreenVelocityAuxFunction::Error: must be a "
      "three dimensional case");
}

void
TaylorGreenVelocityAuxFunction::do_evaluate(
  const double* coords,
  const double /* t */,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for (unsigned p = 0; p < numPoints; ++p) {

    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];

    fieldPtr[0] = +uNot_ * sin(x / L_) * cos(y / L_) * cos(z / L_);
    fieldPtr[1] = -uNot_ * cos(x / L_) * sin(y / L_) * cos(z / L_);
    fieldPtr[2] = 0.0;
    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace sierra
