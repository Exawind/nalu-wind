/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "user_functions/WindEnergyMomWakeInductionAuxFunction.h"

// basic c++
#include <cmath>
#include <array>

namespace sierra{
namespace nalu{

WindEnergyMomWakeInductionAuxFunction::WindEnergyMomWakeInductionAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double> &params) :
  AuxFunction(beginPos, endPos)
{
  // check size and populate
  if ( params.size() != 8 )
    throw std::runtime_error("Realm::setup_initial_conditions: wind_energy_mom_ind_wake requires 8 params: ");
  turbine_loc_[0] = params[0];
  turbine_loc_[1] = params[1];
  turbine_loc_[2] = params[2];
  ct_ = params[3];
  dia_ = params[4];
  r_ = 0.5 * dia_ ; 
  u_infty_[0] = params[5];
  u_infty_[1] = params[6];
  u_infty_[2] = params[7];
  u_mag_ = std::sqrt(u_infty_[0] * u_infty_[0] + u_infty_[1] * u_infty_[1] + u_infty_[2] * u_infty_[2]);
  for (auto i=0;i<3;i++)
      u_infty_[i] = u_infty_[i]/u_mag_; //Normalize u_infty to store the flow direction
  
}

void
WindEnergyMomWakeInductionAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{

    std::array<double,3> c_rel; // Coordinates relative to turbine
    double rsq;
    double crel_dot_n;
    double u_def;
    double sqrt_one_p_zsq;
    double a = 0.5 * (1.0 - std::sqrt(1.0 - ct_)); // Induction
    double gamma_t = -2.0 * a * u_mag_;
    double r_wake_sq;
    
    for(unsigned p=0; p < numPoints; ++p) {
        

        for (auto i=0; i < 3; i++)
            c_rel[i] = coords[i] - turbine_loc_[i];
        crel_dot_n = 0.0;
        for (auto i=0; i < 3; i++)
            crel_dot_n += c_rel[i] * u_infty_[i];
        rsq = 0.0;
        for (auto i=0; i < 3; i++) {
            c_rel[i] = c_rel[i] - crel_dot_n * u_infty_[i];
            rsq += c_rel[i] * c_rel[i];
        }

        crel_dot_n = crel_dot_n/r_; // Normalize w.r.t turbine radius
        sqrt_one_p_zsq = std::sqrt(1.0 + crel_dot_n * crel_dot_n);
        
        r_wake_sq = r_ * r_ * (1.0 - a)/ (1.0 - a * (1.0 + crel_dot_n/std::sqrt(1.0 + crel_dot_n*crel_dot_n) ) );
        rsq = rsq/r_wake_sq;
        
        u_def = gamma_t * 0.5 * (1.0 + crel_dot_n / sqrt_one_p_zsq ) *   ( 0.5 * (1.0 - std::tanh( (rsq - 1.0) * 100.0 ) ) ) * ( 0.5 * (1.0 - std::tanh( (crel_dot_n - 16.0)*0.5 ) ) );
        
        fieldPtr[0] = (u_mag_ + u_def) * u_infty_[0];
        fieldPtr[1] = (u_mag_ + u_def) * u_infty_[1];
        fieldPtr[2] = (u_mag_ + u_def) * u_infty_[2];
        
        fieldPtr += fieldSize;
        coords += fieldSize;
    }
}

} // namespace nalu
} // namespace Sierra
