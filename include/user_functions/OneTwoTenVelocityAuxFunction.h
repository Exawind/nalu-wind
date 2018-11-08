/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef OneTwoTenVelocityAuxFunction_h
#define OneTwoTenVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class OneTwoTenVelocityAuxFunction : public AuxFunction
{
public:

  OneTwoTenVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~OneTwoTenVelocityAuxFunction() {}
  
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
  const double dpdz_;
  const double mu_;
  const double a_;
  const double b_;
  const int maxN_;
  const double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
