/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BoussinesqNonIsoTemperatureAuxFunction_h
#define BoussinesqNonIsoTemperatureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class BoussinesqNonIsoTemperatureAuxFunction : public AuxFunction
{
public:

  BoussinesqNonIsoTemperatureAuxFunction();

  virtual ~BoussinesqNonIsoTemperatureAuxFunction() {}
  
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
  const double Cp_;
  const double Tref_;
};

} // namespace nalu
} // namespace Sierra

#endif
