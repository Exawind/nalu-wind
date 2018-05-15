/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SinProfileChannelFlowVelocityAuxFunction_h
#define SinProfileChannelFlowVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class SinProfileChannelFlowVelocityAuxFunction : public AuxFunction
{
public:

  SinProfileChannelFlowVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~SinProfileChannelFlowVelocityAuxFunction() {}
  
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
