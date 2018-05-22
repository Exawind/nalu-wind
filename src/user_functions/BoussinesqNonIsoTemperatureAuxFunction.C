/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/BoussinesqNonIsoTemperatureAuxFunction.h>
#include <algorithm>
#include <NaluEnv.h>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

BoussinesqNonIsoTemperatureAuxFunction::BoussinesqNonIsoTemperatureAuxFunction() :
  AuxFunction(0,1),
  Cp_(0.01),
  Tref_(300.0)
{
  // does nothing
}

void
BoussinesqNonIsoTemperatureAuxFunction::do_evaluate(
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
    const double z = coords[2];

    const double h = z;

    const double temp = h / Cp_ + Tref_;

    fieldPtr[0] = temp;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
