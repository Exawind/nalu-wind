// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef WindEnergyTaylorVortexPressureAuxFunction_h
#define WindEnergyTaylorVortexPressureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra {
namespace nalu {

class WindEnergyTaylorVortexPressureAuxFunction : public AuxFunction
{
public:
  WindEnergyTaylorVortexPressureAuxFunction(
    const std::vector<double>& theParams);

  virtual ~WindEnergyTaylorVortexPressureAuxFunction() {}

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
  double centroidX_;
  double centroidY_;
  double rVortex_;
  double beta_;
  double uInf_;
  double density_;
};

class WindEnergyTaylorVortexPressureGradAuxFunction : public AuxFunction
{
public:
  WindEnergyTaylorVortexPressureGradAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    const std::vector<double>& theParams);

  virtual ~WindEnergyTaylorVortexPressureGradAuxFunction() {}

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
  double centroidX_;
  double centroidY_;
  double rVortex_;
  double beta_;
  double uInf_;
  double density_;
  double visc_;
};

} // namespace nalu
} // namespace sierra

#endif
