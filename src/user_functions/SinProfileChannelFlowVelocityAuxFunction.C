// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/SinProfileChannelFlowVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

SinProfileChannelFlowVelocityAuxFunction::
  SinProfileChannelFlowVelocityAuxFunction(
    const unsigned beginPos, const unsigned endPos)
  : AuxFunction(beginPos, endPos), u_m(10.0) /*,		// bulk velocity
                                   pi_(acos(-1.0)) */
{
  // does nothing
}

void
SinProfileChannelFlowVelocityAuxFunction::do_evaluate(
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

    const double aux_x = (y > 1) ? 1.0 : -1.0;

    fieldPtr[0] = u_m * sin(x) * aux_x;
    fieldPtr[1] = 0.1 * u_m * sin(y);
    fieldPtr[2] = 0.1 * u_m * sin(z);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
