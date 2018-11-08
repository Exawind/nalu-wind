/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef BoundaryLayerPerturbationAuxFunction_h
#define BoundaryLayerPerturbationAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

/** Add sinusoidal perturbations to the velocity field.
 *
 *  This function is used as an initial condition, primarily in Atmospheric
 *  Boundary Layer (ABL) flows, to trigger transition to turbulent flow during
 *  ABL precursor simulations.
 */
class BoundaryLayerPerturbationAuxFunction : public AuxFunction
{
public:

  BoundaryLayerPerturbationAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double> &theParams);

  virtual ~BoundaryLayerPerturbationAuxFunction() {}
  
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
  /// Amplitude of perturbations
  double amplitude_;
  double kx_;
  double ky_;
  double thickness_;

  /// Mean velocity field during initialization
  double uInf_;

};

} // namespace nalu
} // namespace Sierra

#endif
