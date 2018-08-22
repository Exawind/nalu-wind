/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <user_functions/KovasznayVelocityAuxFunction.h>
#include <algorithm>
#include <stk_util/util/ReportHandler.hpp>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

KovasznayVelocityAuxFunction::KovasznayVelocityAuxFunction(
  const unsigned beginPos,
  const unsigned endPos)
:   AuxFunction(beginPos,endPos),
    Re_(40.0),
    kx_(2*std::acos(-1.)),
    ky_(2*std::acos(-1.))
{
  ThrowRequireMsg(endPos == 2, "Only 2D for Kovasznay flow");
}

void
KovasznayVelocityAuxFunction::do_evaluate(
  const double *coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  double pi = std::acos(-1.);
  double lambda = 0.5 * Re_ - std::sqrt(0.25 * Re_ * Re_ + 4. * pi * pi);

  for(unsigned p=0; p < numPoints; ++p) {
    const double x = coords[0];
    const double y = coords[1];
    fieldPtr[0] = 1. - std::exp(lambda*x) * std::cos(kx_*y);
    fieldPtr[1] = lambda/(2*pi) * std::exp(lambda*x) * std::sin(ky_*y);
    
    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace Sierra
