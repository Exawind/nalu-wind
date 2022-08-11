// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/RayleighTaylorMixFracAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

RayleighTaylorMixFracAuxFunction::RayleighTaylorMixFracAuxFunction()
  : AuxFunction(0, 1),
    aX_(0.1),
    tX_(1.0),
    yTr_(1.0),
    dTr_(0.20),
    pi_(acos(-1.0))
{
  // does nothing
}

void
RayleighTaylorMixFracAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  const double ymin = yTr_ - dTr_ / 2.0;
  const double ymax = yTr_ + dTr_ / 2.0;

  for (unsigned p = 0; p < numPoints; ++p) {

    const double x = coords[0];
    const double y = coords[1];

    const double dy = -aX_ * cos(2.0 * tX_ * pi_ * x);
    const double yy = y + dy;

    double value = 0.0;
    if (yy < ymin) {
      value = 0.0;
    } else if (yy > ymax) {
      value = 1.0;
    } else {
      value = 1.0 / 2.0 * (1.0 - sin(pi_ * yy / dTr_));
    }

    fieldPtr[0] = value;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
