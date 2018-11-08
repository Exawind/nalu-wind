/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WindEnergyAuxFunction_h
#define WindEnergyAuxFunction_h

#include <AuxFunction.h>

#include <string>
#include <vector>

namespace sierra{
namespace nalu{

class Realm;
template<typename T> class TanhFunction;

class WindEnergyAuxFunction : public AuxFunction
{
public:

  WindEnergyAuxFunction(
    const unsigned beginPos,
    const unsigned endPos,
    std::vector<std::string> theStringParams,
    Realm &realm);

  virtual ~WindEnergyAuxFunction();
  
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

  void setup(const double time);
  void cross_product(double *c, double *u) const;

private:
  double omegaBlend_;
  TanhFunction<double> *tanhFunction_;
  std::vector<double> omegaMM_;
  std::vector<double> centroidMM_;
};

} // namespace nalu
} // namespace Sierra

#endif
