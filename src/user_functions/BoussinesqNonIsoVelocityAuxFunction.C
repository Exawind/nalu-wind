/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/BoussinesqNonIsoVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

BoussinesqNonIsoVelocityAuxFunction::BoussinesqNonIsoVelocityAuxFunction(
  const unsigned beginPos,
  const unsigned endPos) :
  AuxFunction(beginPos, endPos)
{
  // must be 3D
  if ( 3 != endPos )
    throw std::runtime_error("BoussinesqNonIsoVelocityAuxFunction::Error: must be a three dimensional case");
}


void
BoussinesqNonIsoVelocityAuxFunction::do_evaluate(
  const double *coords,
  const double /* t */,
  const unsigned /*spatialDimension*/,
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

    fieldPtr[0] = 0.5*cos(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);
    fieldPtr[1] = -sin(2*M_PI*x)*cos(2*M_PI*y)*sin(2*M_PI*z);
    fieldPtr[2] = 0.5*sin(2*M_PI*x)*sin(2*M_PI*y)*cos(2*M_PI*z);

    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
