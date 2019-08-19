/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "user_functions/WindEnergyPowerLawAuxFunction.h"
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

WindEnergyPowerLawAuxFunction::WindEnergyPowerLawAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double> &params) :
  AuxFunction(beginPos, endPos)
{
  // check size and populate
  if ( params.size() != 8 )
    throw std::runtime_error("Realm::setup_initial_conditions: wind_energy_power_law requires 8 params: ");
  coord_dir_  = int(params[0]);
  y_offset_ = params[1];
  y_ref_ = params[2];
  shear_exp_ = params[3];
  u_ref_[0] = params[4];
  u_ref_[1] = params[5];
  u_ref_[2] = params[6];
  u_mag_ = std::sqrt(u_ref_[0] * u_ref_[0] + u_ref_[1] * u_ref_[1] + u_ref_[2] * u_ref_[2]);
  u_min_ = params[7];
  u_max_ = params[8];
}

void
WindEnergyPowerLawAuxFunction::do_evaluate(
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

    const double y = coords[coord_dir_];

    const double power_law_fn = std::pow( (y - y_offset_)/y_ref_ , shear_exp_ ) * 0.5 * ( std::tanh( (y - y_ref_)) + 1.0) ;

    if ( u_mag_* power_law_fn < u_min_)  {
        fieldPtr[0] = u_ref_[0]/u_mag_ * u_min_ ;
        fieldPtr[1] = u_ref_[1]/u_mag_ * u_min_ ;
        fieldPtr[2] = u_ref_[2]/u_mag_ * u_min_ ;
    }
    else if ( u_mag_* power_law_fn > u_max_)  {
        fieldPtr[0] = u_ref_[0]/u_mag_ * u_max_ ;
        fieldPtr[1] = u_ref_[1]/u_mag_ * u_max_ ;
        fieldPtr[2] = u_ref_[2]/u_mag_ * u_max_ ;
    }
    else {
        fieldPtr[0] = u_ref_[0] * power_law_fn;
        fieldPtr[1] = u_ref_[1] * power_law_fn;
        fieldPtr[2] = u_ref_[2] * power_law_fn;
    }
    
    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
