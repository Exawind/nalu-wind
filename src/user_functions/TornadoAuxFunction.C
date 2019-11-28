// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/TornadoAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

TornadoAuxFunction::TornadoAuxFunction(
  const unsigned beginPos,
  const unsigned endPos) :
  AuxFunction(beginPos, endPos),
  z1_(0.025),
  hNot_(0.41),
  rNot_(0.4),
  uRef_(0.3),
  swirl_(2.0)
{
  // nothing
}


void
TornadoAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {

    double cX = coords[0];
    double cY = coords[1];
    double cZ = coords[2];

    const double fac = std::pow(cZ/z1_, 1.0/7.0);
    
    const double uMag = uRef_*fac;
    const double omega = uMag/rNot_;
    const double uZ = 2.0*hNot_/rNot_*swirl_*uMag;
    
    fieldPtr[0] = -omega*cY;
    fieldPtr[1] = +omega*cX;
    fieldPtr[2] = uZ;
    
    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
