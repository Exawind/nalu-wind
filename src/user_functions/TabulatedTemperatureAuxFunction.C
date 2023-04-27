// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/TabulatedTemperatureAuxFunction.h>
#include <algorithm>
#include <NaluEnv.h>
#include "utils/LinearInterpolation.h"

#include <stk_util/util/ReportHandler.hpp>
// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra::nalu {

TabulatedTemperatureAuxFunction::TabulatedTemperatureAuxFunction(
  std::vector<double> heights, std::vector<double> temperatures)
  : AuxFunction(0, 1), heights_(heights), temperatures_(temperatures)
{
}

void
TabulatedTemperatureAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned dim,
  const unsigned numPoints,
  double* temperatures,
  const unsigned /*fieldSize*/,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for (unsigned p = 0; p < numPoints; ++p) {
    const double height = coords[dim * p + 2];
    utils::linear_interp(heights_, temperatures_, height, temperatures[p]);
  }
}

} // namespace sierra::nalu
