// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#include <user_functions/CappingInversionTemperatureAuxFunction.h>
#include <algorithm>
#include <NaluEnv.h>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

CappingInversionTemperatureAuxFunction::CappingInversionTemperatureAuxFunction(
  const std::vector<double> &params) :
  AuxFunction(0,1)
{
  // check size and populate
  if ( params.size() != 5 )
    throw std::runtime_error("Realm::setup_initial_conditions: capping_inversion requires 5 params: ");
  T_belowCap_ = params[0]; 
  T_aboveCap_ = params[1]; 
  weakInversionStrength_ = params[2];
  z_bottomCap_ = params[3];
  z_topCap_ = params[4];
}

void
CappingInversionTemperatureAuxFunction::do_evaluate(
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

    const double slope_1 = (T_aboveCap_ - T_belowCap_) / (z_topCap_ - z_bottomCap_);

    double temp = T_belowCap_;
    if ( z > z_bottomCap_ && z <= z_topCap_ ) {
      temp = T_belowCap_ + slope_1*(z-z_bottomCap_);
    }
    else if ( z > z_topCap_ ) {
      temp = T_aboveCap_ + weakInversionStrength_*(z-z_topCap_);
    }
      
    fieldPtr[0] = temp;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
