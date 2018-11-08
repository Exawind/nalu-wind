/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ConvectingTaylorVortexPressureAuxFunction_h
#define ConvectingTaylorVortexPressureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class ConvectingTaylorVortexPressureAuxFunction : public AuxFunction
{
public:

  ConvectingTaylorVortexPressureAuxFunction();

  virtual ~ConvectingTaylorVortexPressureAuxFunction() {}
  
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
  double uNot_;
  double vNot_;
  double pNot_;
  double visc_;
  double pi_;

};

class ConvectingTaylorVortexPressureGradAuxFunction : public AuxFunction
{
public:

  ConvectingTaylorVortexPressureGradAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~ConvectingTaylorVortexPressureGradAuxFunction() {}

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
  double uNot_;
  double vNot_;
  double pNot_;
  double visc_;
  double pi_;

};

} // namespace nalu
} // namespace Sierra

#endif
