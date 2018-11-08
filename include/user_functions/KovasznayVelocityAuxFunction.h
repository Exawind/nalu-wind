/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef KovasznayVelocityAuxFunction_h
#define KovasznayVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class KovasznayVelocityAuxFunction : public AuxFunction
{
public:

  KovasznayVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~KovasznayVelocityAuxFunction() {}
  
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
  double Re_;
  double kx_;
  double ky_;

};

} // namespace nalu
} // namespace Sierra

#endif
