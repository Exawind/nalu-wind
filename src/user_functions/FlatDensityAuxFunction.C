// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/FlatDensityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace sierra {
namespace nalu {

FlatDensityAuxFunction::FlatDensityAuxFunction() : AuxFunction(0, 1)
{
  // does nothing
}

void
FlatDensityAuxFunction::do_evaluate(
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
    const double interface_thickness = 0.015;

    const double vof_function =
      -0.5 * (std::erf(y / interface_thickness) + 1.0) + 1.0;

    // air-water
    fieldPtr[0] = 1000.0 * vof_function + (1.0 - vof_function);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
