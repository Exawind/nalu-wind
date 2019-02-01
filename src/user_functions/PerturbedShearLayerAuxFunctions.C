/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/PerturbedShearLayerAuxFunctions.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>
#include <random>

namespace sierra{
namespace nalu{

PerturbedShearLayerVelocityAuxFunction::PerturbedShearLayerVelocityAuxFunction(
  const unsigned beginPos,
  const unsigned endPos) :
  AuxFunction(beginPos, endPos)
{
  if (endPos != 3)
    throw std::runtime_error("3 dimensional only");
}

namespace {

double wrap_value(double value, double wrap )
{
    return value - wrap * floor(value / wrap);
}

class ShearLayerHelper
{
public:
  double funu(double y) {
    const double ymod = (do_wrap) ? wrap_value(y, 4 * M_PI) : y;

    if (do_wrap) {
      return (std::tanh(inv_initial_vorticity_thickness * (ymod - 2 * M_PI))
      - std::tanh(inv_initial_vorticity_thickness * (ymod - 4 * M_PI))
      - std::tanh(inv_initial_vorticity_thickness * ymod));
    }
    return std::tanh(inv_initial_vorticity_thickness * ymod);
  }

  double funvw(double y)
  {
    const double ymod = (do_wrap) ? wrap_value(y + M_PI, 2 * M_PI) : (y + M_PI);
    return perturb_mag * std::exp(-100/(size_ratio_x*size_ratio_x) * (ymod-M_PI)*(ymod-M_PI));
  }


  const double size_ratio_x{4 * M_PI};
  const double size_ratio_y{16/3.0 * M_PI};
  const double size_ratio_z{12 * M_PI};
  const double perturb_mag{0.1};
  const double inv_initial_vorticity_thickness{10/size_ratio_x};
  const bool do_wrap{false};
};

}


void
PerturbedShearLayerVelocityAuxFunction::do_evaluate(
  const double *coords,
  const double /* t */,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned  /* fieldSize */,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed); // fixed seed
  std::uniform_real_distribution<double> r1(-0.05, 0.05);

  const double kx = 2;
  const double kz = 32;
  ShearLayerHelper slh;

  for (unsigned p = 0; p < numPoints; ++p) {
    const double x = coords[3 * p + 0] * slh.size_ratio_x;
    const double y = coords[3 * p + 1] * slh.size_ratio_y;
    const double z = coords[3 * p + 2] * slh.size_ratio_z;
    fieldPtr[3 * p + 0] = slh.funu(y) + slh.funvw(y) * (std::sin(2 * kx * x) + 0.01 * std::cos(kz * z) + r1(rng));
    fieldPtr[3 * p + 1] =               slh.funvw(y) * (std::sin(kx * x) + 0.01 * std::cos(2 * kz * z) + r1(rng));
    fieldPtr[3 * p + 2] =               slh.funvw(y) * (std::cos(2 * kx * x) + 0.01 * std::sin(kz * z) + r1(rng));
  }
}

PerturbedShearLayerMixFracAuxFunction::PerturbedShearLayerMixFracAuxFunction() : AuxFunction(0,1)
{}

void
PerturbedShearLayerMixFracAuxFunction::do_evaluate(
  const double *coords,
  const double /* t */,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned  /* fieldSize */,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  ShearLayerHelper slh;
  for(unsigned p=0; p < numPoints; ++p) {
    const double y = coords[spatialDimension * p + 1] * slh.size_ratio_y;
    fieldPtr[p] = 0.5*(1-slh.funu(y));
  }
}

} // namespace nalu
} // namespace Sierra
