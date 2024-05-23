// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/DropletVOFAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

DropletVOFAuxFunction::DropletVOFAuxFunction() : AuxFunction(0, 1)
{
  // does nothing
}

void
DropletVOFAuxFunction::do_evaluate(
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
    const double interface_thickness = 0.0025;

    fieldPtr[0] = 0.0;
    fieldPtr[0] += -0.5 * (std::erf(y / interface_thickness) + 1.0) + 1.0;

    auto radius = std::sqrt(x * x + (y - 0.25) * (y - 0.25) + z * z) - 0.1;
    // fieldPtr[0] += -0.5 * (std::erf(radius / interface_thickness) + 1.0)
    // + 1.0;
    fieldPtr[0] += -0.5 * (std::erf(radius / 0.0025) + 1.0) + 1.0;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
