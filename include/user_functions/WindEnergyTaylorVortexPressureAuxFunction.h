/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WindEnergyTaylorVortexPressureAuxFunction_h
#define WindEnergyTaylorVortexPressureAuxFunction_h

#include <AuxFunction.h>

#include <vector>

namespace sierra{
namespace nalu{

class WindEnergyTaylorVortexPressureAuxFunction : public AuxFunction
{
public:

  WindEnergyTaylorVortexPressureAuxFunction(const std::vector<double> &theParams);

  virtual ~WindEnergyTaylorVortexPressureAuxFunction() {}

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
    const std::vector<double> &theParams);

  virtual ~WindEnergyTaylorVortexPressureGradAuxFunction() {}

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
  double centroidX_;
  double centroidY_;
  double rVortex_;
  double beta_;
  double uInf_;
  double density_;
  double visc_;

};

} // namespace nalu
} // namespace Sierra

#endif
