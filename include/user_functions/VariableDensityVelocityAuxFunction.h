/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef VariableDensityVelocityAuxFunction_h
#define VariableDensityVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class VariableDensityVelocityAuxFunction : public AuxFunction
{
public:

  VariableDensityVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~VariableDensityVelocityAuxFunction() {}
  
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
  const double wnot_;
  const double a_;
  const double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
