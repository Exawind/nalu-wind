// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/StringTimeCoordTemperatureAuxFunction.h>
#include <algorithm>

namespace sierra::nalu {

StringTimeCoordTemperatureAuxFunction::StringTimeCoordTemperatureAuxFunction(
  std::string fcn)
  : AuxFunction(0, 1), f_(fcn)
{
}

void
StringTimeCoordTemperatureAuxFunction::do_evaluate(
  const double* coords,
  const double time,
  const unsigned dim,
  const unsigned numPoints,
  double* temperatures,
  const unsigned /*fieldSize*/,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  if (f_.spatial_dim() > int(dim)) {
    // e.g. if someone has "z" coord in 2D
    throw std::runtime_error("Dimensional arguments to string function greater "
                             "than simulation dimension");
  }

  if (dim == 3) {
    for (unsigned p = 0; p < numPoints; ++p) {
      temperatures[p] =
        f_(time, coords[3 * p + 0], coords[3 * p + 1], coords[3 * p + 2]);
    }
  } else if (dim == 2) {
    for (unsigned p = 0; p < numPoints; ++p) {
      temperatures[p] = f_(time, coords[2 * p + 0], coords[2 * p + 1], 0);
    }
  }
}

} // namespace sierra::nalu
