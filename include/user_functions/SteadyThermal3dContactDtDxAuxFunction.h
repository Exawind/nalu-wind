/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SteadyThermal3dContactDtDxAuxFunction_h
#define SteadyThermal3dContactDtDxAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class SteadyThermal3dContactDtDxAuxFunction : public AuxFunction
{
public:

  SteadyThermal3dContactDtDxAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~SteadyThermal3dContactDtDxAuxFunction() {}
  
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
  double a_;
  double k_;
  double pi_;

};

} // namespace nalu
} // namespace Sierra

#endif
