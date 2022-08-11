// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/OneTwoTenVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

OneTwoTenVelocityAuxFunction::OneTwoTenVelocityAuxFunction(
  const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos),
    dpdz_(-0.0016),
    mu_(1.0e-4),
    a_(0.5),
    b_(1.0),
    maxN_(100),
    pi_(acos(-1.0))
{
  // does nothing
}

void
OneTwoTenVelocityAuxFunction::do_evaluate(
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

    double sum = 0.0;
    for (int n = 0; n < maxN_; ++n) {
      const double m = (2.0 * n + 1.0) * pi_ / 2.0 / b_;
      sum += std::pow(-1, n) * 1.0 / m / m / m *
             (std::cos(m * y) * std::cosh(m * x)) / std::cosh(m * a_);
    }
    const double w =
      -1.0 / 2.0 / mu_ * dpdz_ * (b_ * b_ - y * y - 4 / b_ * sum);

    fieldPtr[0] = 0.0;
    fieldPtr[1] = 0.0;
    fieldPtr[2] = w;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
