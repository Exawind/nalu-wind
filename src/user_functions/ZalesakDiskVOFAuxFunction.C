// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/ZalesakDiskVOFAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

ZalesakDiskVOFAuxFunction::ZalesakDiskVOFAuxFunction() : AuxFunction(0, 1)
{
  // does nothing
}

void
ZalesakDiskVOFAuxFunction::do_evaluate(
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

    fieldPtr[0] = 0.0;
    if (x * x + (y - 0.25) * (y - 0.25) < 0.15 * 0.15)
      fieldPtr[0] = 1.0;

    if (x > -0.025 && x < 0.025 && y < 0.35)
      fieldPtr[0] = 0.0;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
