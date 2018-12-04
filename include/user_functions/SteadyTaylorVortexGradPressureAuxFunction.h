/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SteadyTaylorVortexGradPressureAuxFunction_h
#define SteadyTaylorVortexGradPressureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class SteadyTaylorVortexGradPressureAuxFunction : public AuxFunction
{
public:

  SteadyTaylorVortexGradPressureAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~SteadyTaylorVortexGradPressureAuxFunction() {}
  
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
  const double a_;
  const double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
