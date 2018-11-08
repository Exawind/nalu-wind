/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef FlowPastCylinderTempAuxFunction_h
#define FlowPastCylinderTempAuxFunction_h

#include <AuxFunction.h>

namespace sierra{
namespace nalu{

class FlowPastCylinderTempAuxFunction : public AuxFunction
{
public:
  FlowPastCylinderTempAuxFunction();

  virtual ~FlowPastCylinderTempAuxFunction() {}
  
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
  
  int find_index( const double z, int iMin, int iMax) const;
  double interpolate_data( const double z) const;
  double local_interpolation( const double z, const int index0, const int index1) const;

private:
  double h_;
  double k_;
  double pi_;
  double experimentalData_[25][2];
  int iMin_;
  int iMax_;
};

} // namespace nalu
} // namespace Sierra

#endif
