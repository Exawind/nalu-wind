// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/WindEnergyTaylorVortexAuxFunction.h>
#include <NaluEnv.h>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace sierra {
namespace nalu {

WindEnergyTaylorVortexAuxFunction::WindEnergyTaylorVortexAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double>& params)
  : AuxFunction(beginPos, endPos),
    centroidX_(-2.5),
    centroidY_(0.0),
    rVortex_(0.25),
    beta_(15.0),
    uInf_(10.0),
    density_(1.0e-3),
    visc_(1.0e-4)
{
  //  check size and populate
  if (!(params.size() > 4 && params.size() < 8) && !params.empty())
    throw std::runtime_error(
      "Realm::setup_initial_conditions: wind_energy_taylor_vortex takes 5 - 7 "
      "parameters:"
      " centroidX, centroidY, initial vortex radius, utheta0, uInf,  "
      "Optionally, density, and viscosity");

  if (!params.empty()) {
    centroidX_ = params[0];
    centroidY_ = params[1];
    rVortex_ = params[2];
    beta_ = params[3];
    uInf_ = params[4];

    if (params.size() > 5) {
      density_ = params[5];
    }

    if (params.size() > 6) {
      visc_ = params[6];
    }
  } else {
    NaluEnv::self().naluOutputP0()
      << "wind_energy_taylor_vortex proceeding with default parameters."
      << "\n  centroidX: " << centroidX_ << "\n  centroidY: " << centroidY_
      << "\n  rVortex: " << rVortex_ << "\n  beta: " << beta_
      << "\n  uinf: " << uInf_ << "\n density: " << density_
      << "\n visc: " << visc_ << std::endl;
  }
}

void
WindEnergyTaylorVortexAuxFunction::do_evaluate(
  const double* coords,
  const double time,
  const unsigned /*spatialDimension*/,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  const double tHat = (visc_ > 1.0e-30)
                        ? time * beta_ / rVortex_
                        : 0.0;               // zero out viscous terms if mu ~ 0
  const double Re = (visc_ > 1.0e-30) ? density_ * beta_ * rVortex_ / visc_
                                      : 0.0; // zero out viscous terms if mu ~ 0
  const double tFac = 1.0 + 2.0 * tHat / Re;
  for (unsigned p = 0; p < numPoints; ++p) {
    const double cX = coords[0];
    const double cY = coords[1];
    const double cDx = cX - (centroidX_ + uInf_ * time);
    const double cDy = cY - centroidY_;

    const double radius = std::sqrt(cDx * cDx + cDy * cDy) / (rVortex_);
    const double factor = beta_ / (rVortex_ * tFac * tFac) *
                          std::exp(0.5 * (1.0 - radius * radius / tFac));

    const double velX = -factor * cDy + uInf_;
    const double velY = factor * cDx;

    fieldPtr[0] = velX;
    fieldPtr[1] = velY;
    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace sierra
