// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <cmath>

#include "actuator/UtilitiesActuator.h"

// This is to access sierra::nalu::Coordinates
#include "NaluParsing.h"


namespace sierra{
namespace nalu {
namespace actuator_utils {

const double pi = M_PI;
///
/// A Gaussian projection function
///
double Gaussian_projection(
  const int &nDim,  // The dimension of the Gaussian (2 or 3)
  double *dis,      // The distance from the center of the Gaussian
  const Coordinates &epsilon  // The width of the Gaussian
  )
{
  // Compute the force projection weight at this location using a
  // Gaussian function.
  double g;
  if ( nDim == 2 )
    g = (1.0 / (epsilon.x_ * epsilon.y_ * pi)) *
        exp(-pow((dis[0]/epsilon.x_),2.0)
            -pow((dis[1]/epsilon.y_),2.0)
           );
  else
    g = (1.0 / (epsilon.x_ * epsilon.y_ * epsilon.z_ * std::pow(pi,1.5))) *
        exp(-pow((dis[0]/epsilon.x_),2.0)
            -pow((dis[1]/epsilon.y_),2.0)
            -pow((dis[2]/epsilon.z_),2.0)
           );

  return g;
}
///
/// A Gaussian projection function
///
double Gaussian_projection(
  const int &nDim, 
  double *dis,
  double *epsilon)
{
  // Compute the force projection weight at this location using a
  // Gaussian function.
  double g;
  if ( nDim == 2 )
    g = (1.0 / (epsilon[0] * epsilon[1] * pi)) *
        exp(-pow((dis[0]/epsilon[0]),2.0)
            -pow((dis[1]/epsilon[1]),2.0)
           );
  else
    g = (1.0 / (epsilon[0] * epsilon[1] * epsilon[2] * std::pow(pi,1.5))) *
        exp(-pow((dis[0]/epsilon[0]),2.0)
            -pow((dis[1]/epsilon[1]),2.0)
            -pow((dis[2]/epsilon[2]),2.0)
           );

  return g;
}

}
}
}
