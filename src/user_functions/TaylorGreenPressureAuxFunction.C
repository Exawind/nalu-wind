// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/TaylorGreenPressureAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

TaylorGreenPressureAuxFunction::TaylorGreenPressureAuxFunction()
  : AuxFunction(0, 1), uNot_(1.0), pNot_(1.0), rhoNot_(1.0), L_(1.0)
{
  // nothing to do
}

void
TaylorGreenPressureAuxFunction::do_evaluate(
  const double* coords,
  const double /* t */,
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

    fieldPtr[0] = pNot_ + rhoNot_ * uNot_ * uNot_ / 16.0 *
                            (cos(2.0 * x / L_) + cos(2.0 * y / L_)) *
                            (cos(2.0 * z / L_) + 2.0);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
