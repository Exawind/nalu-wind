// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/VariableDensityNonIsoTemperatureAuxFunction.h>
#include <algorithm>
#include <NaluEnv.h>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

VariableDensityNonIsoTemperatureAuxFunction::VariableDensityNonIsoTemperatureAuxFunction() :
  AuxFunction(0,1),
  ah_(10.0),
  Cp_(0.01),
  Tref_(300.0),
  pi_(acos(-1.0))
{
  // does nothing
}

void
VariableDensityNonIsoTemperatureAuxFunction::do_evaluate(
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
    
    const double h = cos(ah_*pi_*x)*cos(ah_*pi_*y)*cos(ah_*pi_*z);

    const double temp = h/Cp_ + Tref_;

    fieldPtr[0] = temp;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
