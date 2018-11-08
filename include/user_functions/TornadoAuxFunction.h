/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TornadoAuxFunction_h
#define TornadoAuxFunction_h

#include <AuxFunction.h>

namespace sierra{
namespace nalu{

class TornadoAuxFunction : public AuxFunction
{
public:

  TornadoAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~TornadoAuxFunction() {}
  
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
  const double z1_, hNot_, rNot_, uRef_, swirl_;
};

} // namespace nalu
} // namespace Sierra

#endif
