// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/DropletVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

DropletVelocityAuxFunction::DropletVelocityAuxFunction(
  const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos)
{
  // does nothing
}

void
DropletVelocityAuxFunction::do_evaluate(
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

    auto radius = std::sqrt(x * x + y * y + z * z) - 0.075;
    auto vof = -0.5 * (std::erf(radius / interface_thickness) + 1.0) + 1.0;
    // assuming density ratio of 1000
    auto dens = 1000.0 * vof + 1.0 * (1.0 - vof);

    fieldPtr[0] = 0.0; // vof * 1000. * 1. / dens;
    fieldPtr[1] = 0.0; // vof * 1000. * 1. / dens;
    fieldPtr[2] = 0.0; // vof * 1000. * 1. / dens;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
