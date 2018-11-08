/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SteadyTaylorVortexVelocityAuxFunction_h
#define SteadyTaylorVortexVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class SteadyTaylorVortexVelocityAuxFunction : public AuxFunction
{
public:

  SteadyTaylorVortexVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~SteadyTaylorVortexVelocityAuxFunction() {}
  
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
  const double unot_;
  const double vnot_;
  const double a_;
  const double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
