/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/SteadyThermalContactAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

SteadyThermalContactAuxFunction::SteadyThermalContactAuxFunction() :
  AuxFunction(0,1),
  a_(1.0),
  k_(1.0),
  pi_(std::acos(-1.0))
{
  // nothing to do
}


void
SteadyThermalContactAuxFunction::do_evaluate(
  const double *coords,
  const double /* t */,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {

    double x = coords[0];
    double y = coords[1];

    fieldPtr[0] = k_/4.0*(cos(2.*a_*pi_*x) + cos(2.*a_*pi_*y));
    
    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
