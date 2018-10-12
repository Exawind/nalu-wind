/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/GaussianWakeVelocityAuxFunction.h>

// basic c++
#include <cmath>
#include <algorithm>

#include <stdexcept>

namespace sierra{
namespace nalu{

GaussianWakeVelocityAuxFunction::GaussianWakeVelocityAuxFunction(const std::vector<double> &params) :
  AuxFunction(0, 3)
{
  //  check size and populate
   if ( params.size() != 7 ) {
     throw std::runtime_error("Realm::setup_initial_conditions: gaussian_wake takes 7 parameters:"
         " centroidX, centroidY, centroidZ, u0, r0, C_T");
   }

   xc_ = params[0];
   yc_ = params[1];
   zc_ = params[2];
   u0_ = params[3];
   r0_ = params[4];
   thrustCoeff_ = params[5];
   alpha_ = params[6];

}

double GaussianWakeVelocityAuxFunction::sigma(double x) const { return alpha_ * std::pow(std::abs(x), 1.0/3.0) + 1; }

double GaussianWakeVelocityAuxFunction::axial_coefficient(double x) const
{
  const double c0 = 1 - std::sqrt(1 - thrustCoeff_);
  return (1 - std::sqrt(1 - c0 * (2 - c0) / (sigma(x) * sigma(x))));
}

void
GaussianWakeVelocityAuxFunction::do_evaluate(
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

    double x = (coords[0] - xc_) / r0_;
    double y = (coords[1] - yc_) / r0_;
    double z = (coords[2] - zc_) / r0_;
    
    double cyl_radsq = y * y + z * z;

    fieldPtr[0] = (x < 1.0e-12) ? u0_ * (1 - axial_coefficient(-x) * std::exp( - cyl_radsq / (2 * sigma(-x) * sigma(-x)))) : u0_;
    fieldPtr[1] = 0;
    fieldPtr[2] = 0;
    
    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
