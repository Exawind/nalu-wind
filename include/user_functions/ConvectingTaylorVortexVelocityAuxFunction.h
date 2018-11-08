/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ConvectingTaylorVortexVelocityAuxFunction_h
#define ConvectingTaylorVortexVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class ConvectingTaylorVortexVelocityAuxFunction : public AuxFunction
{
public:

  ConvectingTaylorVortexVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~ConvectingTaylorVortexVelocityAuxFunction() {}
  
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
  double uNot_;
  double vNot_;
  double visc_;
  double pi_;

};

} // namespace nalu
} // namespace Sierra

#endif
