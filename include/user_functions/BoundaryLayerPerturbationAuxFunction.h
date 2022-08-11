// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef BoundaryLayerPerturbationAuxFunction_h
#define BoundaryLayerPerturbationAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra {
namespace nalu {

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
    const std::vector<double>& theParams);

  virtual ~BoundaryLayerPerturbationAuxFunction() {}

  using AuxFunction::do_evaluate;
  virtual void do_evaluate(
    const double* coords,
    const double time,
    const unsigned spatialDimension,
    const unsigned numPoints,
    double* fieldPtr,
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
} // namespace sierra

#endif
