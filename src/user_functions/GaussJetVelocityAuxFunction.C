// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/GaussJetVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

GaussJetVelocityAuxFunction::GaussJetVelocityAuxFunction(
  const unsigned beginPos,
  const unsigned endPos) :
  AuxFunction(beginPos, endPos),
    u_m(10.0) 	// bulk velocity
{
  // does nothing
}

void
GaussJetVelocityAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  if(spatialDimension == 2)
  {
    for(unsigned p=0; p < numPoints; ++p) {

      const double x = coords[0];

      fieldPtr[0] = 0.0;
      fieldPtr[1] = -u_m*exp(-10.0*x*x);
      fieldPtr += fieldSize;
      coords += spatialDimension;
    }
  }
  else{
    for(unsigned p=0; p < numPoints; ++p) {

      const double x = coords[0];
      const double y = coords[1];
      const double r2 = x*x+y*y;

      fieldPtr[0] = 0.0;
      fieldPtr[1] = 0.0;
      fieldPtr[2] = -u_m*exp(-10.0*r2);

      fieldPtr += fieldSize;
      coords += spatialDimension;
    }
  }
}

} // namespace nalu
} // namespace Sierra
