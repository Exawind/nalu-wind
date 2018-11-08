/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WindEnergyTaylorVortexAuxFunction_h
#define WindEnergyTaylorVortexAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class WindEnergyTaylorVortexAuxFunction : public AuxFunction
{
public:

  WindEnergyTaylorVortexAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double> &theParams);

  virtual ~WindEnergyTaylorVortexAuxFunction() {}
  
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
  double centroidX_;
  double centroidY_;
  double rVortex_;
  double beta_;
  double uInf_;
  double density_;
  double visc_;
};

} // namespace nalu
} // namespace Sierra

#endif
