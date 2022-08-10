// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/VariableDensityVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

VariableDensityVelocityAuxFunction::VariableDensityVelocityAuxFunction(
  const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos),
    unot_(1.0),
    vnot_(1.0),
    wnot_(1.0),
    a_(20.0),
    pi_(acos(-1.0))
{
  // does nothing
}

void
VariableDensityVelocityAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned spatialDimension,
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

    fieldPtr[0] =
      -unot_ * cos(a_ * pi_ * x) * sin(a_ * pi_ * y) * sin(a_ * pi_ * z);
    fieldPtr[1] =
      +vnot_ * sin(a_ * pi_ * x) * cos(a_ * pi_ * y) * sin(a_ * pi_ * z);
    fieldPtr[2] =
      -wnot_ * sin(a_ * pi_ * x) * sin(a_ * pi_ * y) * cos(a_ * pi_ * z);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
