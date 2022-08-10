// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/KovasznayPressureAuxFunction.h>
#include <algorithm>
#include <stk_util/util/ReportHandler.hpp>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

KovasznayPressureAuxFunction::KovasznayPressureAuxFunction()
  : AuxFunction(0, 1), Re_(40.0) /*,
                       kx_(2*std::acos(-1.)),
                       ky_(2*std::acos(-1.)) */
{
}

void
KovasznayPressureAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned /*fieldSize*/,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  double pi = std::acos(-1.);
  double lambda = 0.5 * Re_ - std::sqrt(0.25 * Re_ * Re_ + 4. * pi * pi);

  for (unsigned p = 0; p < numPoints; ++p) {
    const double x = coords[0];
    fieldPtr[0] = 0.5 * (1. - std::exp(2 * lambda * x));

    fieldPtr += 1;
    coords += 2;
  }
}

KovasznayPressureGradientAuxFunction::KovasznayPressureGradientAuxFunction(
  const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos), Re_(40.0) /*,
                                   kx_(2*std::acos(-1.)),
                                   ky_(2*std::acos(-1.)) */
{
}

void
KovasznayPressureGradientAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned /*fieldSize*/,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  double pi = std::acos(-1.);
  double lambda = 0.5 * Re_ - std::sqrt(0.25 * Re_ * Re_ + 4. * pi * pi);

  for (unsigned p = 0; p < numPoints; ++p) {
    const double x = coords[0];
    fieldPtr[0] = -lambda * std::exp(2 * lambda * x);
    fieldPtr[1] = 0;

    fieldPtr += 2;
    coords += 2;
  }
}

} // namespace nalu
} // namespace sierra
