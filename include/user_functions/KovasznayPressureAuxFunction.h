/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef KovasznayPressureAuxFunction_h
#define KovasznayPressureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class KovasznayPressureAuxFunction : public AuxFunction
{
public:

  KovasznayPressureAuxFunction();

  virtual ~KovasznayPressureAuxFunction() {}
  
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
};

class KovasznayPressureGradientAuxFunction : public AuxFunction
{
public:

  KovasznayPressureGradientAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~KovasznayPressureGradientAuxFunction() {}

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
};

} // namespace nalu
} // namespace Sierra

#endif
