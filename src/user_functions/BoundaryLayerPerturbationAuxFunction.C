// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/BoundaryLayerPerturbationAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

BoundaryLayerPerturbationAuxFunction::BoundaryLayerPerturbationAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double>& params)
  : AuxFunction(beginPos, endPos),
    amplitude_(0.05),
    kx_(0.1),
    ky_(0.1),
    thickness_(0.05),
    uInf_(10.0)
{
  // check size and populate
  if (params.size() != 5)
    throw std::runtime_error(
      "Realm::setup_initial_conditions: "
      "boundary_layer_perturbation requires 5 params: ");
  amplitude_ = params[0];
  kx_ = params[1];
  ky_ = params[2];
  thickness_ = params[3];
  uInf_ = params[4];
}

void
BoundaryLayerPerturbationAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for (unsigned p = 0; p < numPoints; ++p) {

    const double cX = coords[0];
    const double cY = coords[1];
    const double cZ = coords[2];

    const double dampfun =
      std::exp(-cZ / thickness_) * cZ / thickness_ / std::exp(-1.0);
    const double Upower = std::pow((cZ / (5.0 * thickness_)), 1.0 / 7.0);
    const double Umean = std::min(Upower, 1.0) * uInf_;

    const double velX =
      Umean + amplitude_ * std::cos(kx_ * cX) * std::cos(ky_ * cY) * dampfun;
    const double velY = amplitude_ * kx_ / ky_ * std::sin(kx_ * cX) *
                        std::sin(ky_ * cY) * dampfun;
    const double velZ = 0.0;

    fieldPtr[0] = velX;
    fieldPtr[1] = velY;
    fieldPtr[2] = velZ;

    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace sierra
