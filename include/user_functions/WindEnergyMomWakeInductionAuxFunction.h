/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WINDENERGYMOMWAKEINDUCTIONAUXFUNCTION_H
#define WINDENERGYMOMWAKEINDUCTIONAUXFUNCTION_H

#include "AuxFunction.h"
#include <vector>
#include <array>

namespace sierra{
namespace nalu{



/** Create power law velocity profile aux function for wind energy applications
 *
 *  This function is used as an initial condition, primarily in
 *  simulation of wind turbines under uniform inflow. It sets up the
 *  velocity field corresponding to the induction and the wake region
 *  using the simple model based on 1D momentum theory from
 *  E. Branlard and M. Gaunaa, Cylindrical vortex wake model: right
 *  cylinder, Wind Energy, 18, 11, Nov. 2015.
 *  
. The function implemented is 
 * \f[
 *  u = u - u_z
 *  \bar{z} = z/R
 *  u_z = \frac{\gamma}{2} \left ( 1 + \frac{\bar{z}}{ \sqrt{1 + \bar{z}^2} } \right )
 *  \left ( \frac{R_w(\bar{z})}{R} \right )^2 = \frac{1 - a}{1 - a \left ( 1 + \frac{\bar{z}}{ \sqrt{1+\bar{z}^2 } } \right )}
 * \f]
 */
class WindEnergyMomWakeInductionAuxFunction : public AuxFunction
{
public:

  WindEnergyMomWakeInductionAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double> &theParams);

  virtual ~WindEnergyMomWakeInductionAuxFunction() {}
  
  using AuxFunction::do_evaluate;
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;
  
private:

    //Turbine location
    std::array<double,3> turbine_loc_;
    double ct_; //Thrust coefficient
    double dia_; //Turbine diameter
    double r_; //Turbine radius
    // Velocity vector at reference height
    std::array<double,3> u_infty_;
    double u_mag_; // Velocity magnitude
};

} // namespace nalu
} // namespace Sierra


#endif /* WINDENERGYMOMWAKEINDUCTIONAUXFUNCTION_H */
