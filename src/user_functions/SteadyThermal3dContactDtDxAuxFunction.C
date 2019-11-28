// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/SteadyThermal3dContactDtDxAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

SteadyThermal3dContactDtDxAuxFunction::SteadyThermal3dContactDtDxAuxFunction(
  const unsigned beginPos,
  const unsigned endPos) 
  : AuxFunction(beginPos, endPos),
    a_(1.0),
    k_(1.0),
    pi_(acos(-1.0))
{
  // does nothing
}

void
SteadyThermal3dContactDtDxAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {

    const double x = coords[0];
    const double y = coords[1];
    const double z = coords[2];

    const double fac = 2.0*a_*pi_*k_/4.0;
    fieldPtr[0] = -fac*sin(2.0*a_*pi_*x);
    fieldPtr[1] = -fac*sin(2.0*a_*pi_*y);
    fieldPtr[2] = -fac*sin(2.0*a_*pi_*z);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
