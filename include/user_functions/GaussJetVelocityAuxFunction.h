/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef GaussJetVelocityAuxFunction_h
#define GaussJetVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class GaussJetVelocityAuxFunction : public AuxFunction
{
public:

  GaussJetVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~GaussJetVelocityAuxFunction() {}
  
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
  const double u_m;
};

} // namespace nalu
} // namespace Sierra

#endif
