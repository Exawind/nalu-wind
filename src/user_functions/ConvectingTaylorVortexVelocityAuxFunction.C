// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/ConvectingTaylorVortexVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

ConvectingTaylorVortexVelocityAuxFunction::
  ConvectingTaylorVortexVelocityAuxFunction(
    const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos),
    uNot_(1.0),
    vNot_(1.0),
    visc_(0.001),
    pi_(std::acos(-1.0))
{
  // nothing to do
}

void
ConvectingTaylorVortexVelocityAuxFunction::do_evaluate(
  const double* coords,
  const double t,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for (unsigned p = 0; p < numPoints; ++p) {

    double x = coords[0];
    double y = coords[1];
    const double omega = pi_ * pi_ * visc_;

    fieldPtr[0] = uNot_ - cos(pi_ * (x - uNot_ * t)) *
                            sin(pi_ * (y - vNot_ * t)) * exp(-2.0 * omega * t);
    fieldPtr[1] = vNot_ + sin(pi_ * (x - uNot_ * t)) *
                            cos(pi_ * (y - vNot_ * t)) * exp(-2.0 * omega * t);

    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace sierra
