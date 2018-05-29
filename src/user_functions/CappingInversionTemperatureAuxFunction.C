/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/CappingInversionTemperatureAuxFunction.h>
#include <algorithm>
#include <NaluEnv.h>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

CappingInversionTemperatureAuxFunction::CappingInversionTemperatureAuxFunction() :
  AuxFunction(0,1)
{
  // does nothing
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

    
    //heights: [    0, 650.0, 750.0, 1000.0 ]
    //values:  [300.0, 300.0, 308.0,  308.75]


    const double slope_1 = (308.0-300.0)/(750.0-650.0);
    const double slope_2 =(308.75-308.0)/(1000.0-750.0);

    double temp = 300.0;
    if ( z > 650.0 && z <= 750.0 ) {
      temp = 300.0 + slope_1*(z-650.0);
    }
    else if ( z > 750.0 && z <= 1000.0 ) {
      temp = 308.0 + slope_2*(z-750.0);
    }
    else if ( z > 1000.0 ) {
      temp = 308.75;
    }
      
    fieldPtr[0] = temp;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
