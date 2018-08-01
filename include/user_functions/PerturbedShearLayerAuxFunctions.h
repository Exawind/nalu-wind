/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef PerturbedShearLayerVelocityAuxFunction_h
#define PerturbedShearLayerVelocityAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class PerturbedShearLayerVelocityAuxFunction : public AuxFunction
{
public:

  PerturbedShearLayerVelocityAuxFunction(
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~PerturbedShearLayerVelocityAuxFunction() {}
  
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

class PerturbedShearLayerMixFracAuxFunction : public AuxFunction
{
public:

  PerturbedShearLayerMixFracAuxFunction();

  virtual ~PerturbedShearLayerMixFracAuxFunction() {}

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
