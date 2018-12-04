/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef VariableDensityNonIsoTemperatureAuxFunction_h
#define VariableDensityNonIsoTemperatureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class VariableDensityNonIsoTemperatureAuxFunction : public AuxFunction
{
public:

  VariableDensityNonIsoTemperatureAuxFunction();

  virtual ~VariableDensityNonIsoTemperatureAuxFunction() {}
  
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
  const double ah_;
  const double Cp_;
  const double Tref_;
  const double pi_;
};

} // namespace nalu
} // namespace Sierra

#endif
