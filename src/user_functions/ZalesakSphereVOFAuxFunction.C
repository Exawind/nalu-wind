// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/ZalesakSphereVOFAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

ZalesakSphereVOFAuxFunction::ZalesakSphereVOFAuxFunction() : AuxFunction(0, 1)
{
  // does nothing
}

void
ZalesakSphereVOFAuxFunction::do_evaluate(
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

    // These are the default arguments from the corresponding case in AMR-Wind
    const double xc = 0.5;
    const double yc = 0.72;
    const double zc = 0.24;
    const double radius = 0.16;
    const double depth = 0.2;
    const double width = 0.04;

    fieldPtr[0] = 0.0;
    // Put VOF in sphere
    if (
      (x - xc) * (x - xc) + (y - yc) * (y - yc) + (z - zc) * (z - zc) <
      radius * radius)
      fieldPtr[0] = 1.0;

    // Remove slot
    if (
      x - xc > -0.5 * width && x - xc < 0.5 * width && y - yc > radius - depth)
      fieldPtr[0] = 0.0;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
