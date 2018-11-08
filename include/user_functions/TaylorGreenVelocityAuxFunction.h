/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TaylorGreenVelocityAuxFunction_h
#define TaylorGreenVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class TaylorGreenVelocityAuxFunction : public AuxFunction
{
public:

  TaylorGreenVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~TaylorGreenVelocityAuxFunction() {}
  
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
  double L_;
};

} // namespace nalu
} // namespace Sierra

#endif
