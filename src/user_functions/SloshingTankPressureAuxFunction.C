// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/SloshingTankPressureAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace sierra {
namespace nalu {

SloshingTankPressureAuxFunction::SloshingTankPressureAuxFunction(
  const std::vector<double>& params)
  : AuxFunction(0, 1),
    water_level_(0.),
    amplitude_(0.1),
    kappa_(0.25),
    interface_thickness_(0.1),
    domain_height_(0.5)
{
  // check size and populate
  if (params.size() != 5 && !params.empty())
    throw std::runtime_error(
      "Realm::setup_initial_conditions: sloshing_tank requires 5 params: water "
      "level, amplitude, kappa, interface thickness, and domain height");
  if (!params.empty()) {
    water_level_ = params[0];
    amplitude_ = params[1];
    kappa_ = params[2];
    interface_thickness_ = params[3];
    domain_height_ = params[4];
  }
}

void
SloshingTankPressureAuxFunction::do_evaluate(
  const double* coords,
  const double /*time*/,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double* fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for (unsigned p = 0; p < numPoints; ++p) {

    const double x = coords[0];
    const double y = coords[1];
    const double z0 = coords[2];
    const double z1 = domain_height_;

    // These need to come from elsewhere
    const double g = 9.81;

    const double z_init =
      water_level_ + amplitude_ * std::exp(-kappa_ * (x * x + y * y));

    // Position 1 is top, 0 is current
    // Density is rho_l * VOF + rho_g * (1-VOF)
    // Integral of error function is x * erf(x) + e^(-x^2)/sqrt(pi)
    // Integrate by substitution, where x = (z - z0) / interface_thickness_
    const double x0 = (z0 - z_init) / interface_thickness_;
    const double x1 = (z1 - z_init) / interface_thickness_;
    const double int_factor = interface_thickness_;
    const double int_erf0 =
      (x0 * std::erf(x0) + std::exp(-std::pow(x0, 2)) / std::sqrt(M_PI)) *
      int_factor;
    const double int_erf1 =
      (x1 * std::erf(x1) + std::exp(-std::pow(x1, 2)) / std::sqrt(M_PI)) *
      int_factor;

    const double int_vof0 = 0.5 * z0 - 0.5 * int_erf0;
    const double int_vof1 = 0.5 * z1 - 0.5 * int_erf1;
    const double int_rho0 = 1000.0 * int_vof0 + 1.0 * (z0 - int_vof0);
    const double int_rho1 = 1000.0 * int_vof1 + 1.0 * (z1 - int_vof1);
    // vof_local = -0.5 * (std::erf((z - z_init) / interface_thickness_) + 1.0) + 1.0;

    // g * integral(rho)dz
    fieldPtr[0] = g * (int_rho1 - int_rho0);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
