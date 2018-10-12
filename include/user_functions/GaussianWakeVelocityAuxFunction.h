/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef GaussianWakeVelocityAuxFunction_h
#define GaussianWakeVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class GaussianWakeVelocityAuxFunction : public AuxFunction
{
public:

  GaussianWakeVelocityAuxFunction(const std::vector<double>& params);

  virtual void do_evaluate(
    const double * coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double * fieldPtr,
    const unsigned fieldSize,
    const unsigned beginPos,
    const unsigned endPos) const;
  

  double sigma(double x) const;
  double axial_coefficient(double x) const;


private:
  double xc_;
  double yc_;
  double zc_;
  double u0_;
  double r0_;
  double thrustCoeff_;
  double alpha_;
};

} // namespace nalu
} // namespace Sierra

#endif
