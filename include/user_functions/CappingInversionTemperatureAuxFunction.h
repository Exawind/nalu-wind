// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef CappingInversionTemperatureAuxFunction_h
#define CappingInversionTemperatureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{



/** Create simple capping inversion profile aux function for wind energy applications
 *
 *  This function is used as an initial or boundary condition,
 *  primarily in simulation of wind turbines in the atmospheric
 *  boundary layer with a typical profile of temperature that
 *  includes a mixed region, a strong cap, and a weak inversion 
 *  above.
 */
class CappingInversionTemperatureAuxFunction : public AuxFunction
{
public:

  CappingInversionTemperatureAuxFunction(
    const std::vector<double> &theParams);

  virtual ~CappingInversionTemperatureAuxFunction() {}
  
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
    double T_belowCap_; // Constant temperature below the strong capping inversion.
    double T_aboveCap_; // Temperature at the top of the strong capping inversion.
    double weakInversionStrength_; // Strength of the weak inversion above the strong capping inversion (dT/dz [K/m])
    double z_bottomCap_; // Height of the bottom of the strong capping inversion.
    double z_topCap_; // Height of the top of the strong capping inversion.
};

} // namespace nalu
} // namespace Sierra

#endif
