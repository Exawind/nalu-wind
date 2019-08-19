/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
 * \f[]
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

    int coord_dir_{2}; // Coordinate direction - 0/1/2
    double y_offset_{0.0}; //Offset for coordinate
    double y_ref_{0.0}; // Reference height
    double shear_exp_{0.12} ; // Exponent for power law
    // Velocity vector at reference height
    std::array<double,3> u_ref_{{8.0,0.0,0.0}}; 
    double u_mag_{8.0}; // Velocity magnitude
    double u_min_{4.0}; // Minimum velocity to cut off power law
    double u_max_{12.0}; // Maximum velocity to cut off power law
};

} // namespace nalu
} // namespace Sierra


#endif /* WINDENERGYPOWERLAWAUXFUNCTION_H */
