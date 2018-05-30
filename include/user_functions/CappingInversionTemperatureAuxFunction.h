/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CappingInversionTemperatureAuxFunction_h
#define CappingInversionTemperatureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class CappingInversionTemperatureAuxFunction : public AuxFunction
{
public:

  CappingInversionTemperatureAuxFunction();

  virtual ~CappingInversionTemperatureAuxFunction() {}
  
  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;  
};

} // namespace nalu
} // namespace Sierra

#endif
