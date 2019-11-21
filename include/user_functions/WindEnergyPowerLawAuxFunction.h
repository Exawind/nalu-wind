// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef WINDENERGYPOWERLAWAUXFUNCTION_H
#define WINDENERGYPOWERLAWAUXFUNCTION_H

#include "AuxFunction.h"
#include <vector>
#include <array>

namespace sierra{
namespace nalu{



/** Create power law velocity profile aux function for wind energy applications
 *
 *  This function is used as an initial or boundary condition,
 *  primarily in simulation of wind turbines under specified shear
 *  with a power law profile. The function implemented is 
 * \f[
 *  u = u_{ref} \left ( \frac{y - y_{offset}}{y_{ref}} \right )^{shear_exp} \frac{1}{2} \left ( tanh (y - y_{offset}) + 1 \right )
 * \f]
 */
class WindEnergyPowerLawAuxFunction : public AuxFunction
{
public:

  WindEnergyPowerLawAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double> &theParams);

  virtual ~WindEnergyPowerLawAuxFunction() {}
  
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

    int coord_dir_; // Coordinate direction - 0/1/2
    double y_offset_; //Offset for coordinate
    double y_ref_; // Reference height
    double shear_exp_ ; // Exponent for power law
    // Velocity vector at reference height
    std::array<double,3> u_ref_; 
    double u_mag_; // Velocity magnitude
    double u_min_; // Minimum velocity to cut off power law
    double u_max_; // Maximum velocity to cut off power law
};

} // namespace nalu
} // namespace Sierra


#endif /* WINDENERGYPOWERLAWAUXFUNCTION_H */
