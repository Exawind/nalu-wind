// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "user_functions/WindEnergyPowerLawAuxFunction.h"

// basic c++
#include <cmath>
#include <stdexcept>

namespace sierra {
namespace nalu {

WindEnergyPowerLawAuxFunction::WindEnergyPowerLawAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double>& params)
  : AuxFunction(beginPos, endPos)
{
  // check size and populate
  if (params.size() != 9)
    throw std::runtime_error(
      "Realm::setup_initial_conditions: "
      "wind_energy_power_law requires 9 params: ");
  coord_dir_ = int(params[0]);
  y_offset_ = params[1];
  y_ref_ = params[2];
  shear_exp_ = params[3];
  u_ref_[0] = params[4];
  u_ref_[1] = params[5];
  u_ref_[2] = params[6];
  u_mag_ = std::sqrt(
    u_ref_[0] * u_ref_[0] + u_ref_[1] * u_ref_[1] + u_ref_[2] * u_ref_[2]);
  u_min_ = params[7] / u_mag_;
  u_max_ = params[8] / u_mag_;
}

void
WindEnergyPowerLawAuxFunction::do_evaluate(
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

    const double y = coords[coord_dir_];

    double power_law_fn = 0.0;

    if ((y - y_offset_) > 0.0) {
      power_law_fn = std::pow((y - y_offset_) / y_ref_, shear_exp_);
    }

    if (power_law_fn < u_min_) {
      fieldPtr[0] = u_ref_[0] * u_min_;
      fieldPtr[1] = u_ref_[1] * u_min_;
      fieldPtr[2] = u_ref_[2] * u_min_;
    } else if (power_law_fn > u_max_) {
      fieldPtr[0] = u_ref_[0] * u_max_;
      fieldPtr[1] = u_ref_[1] * u_max_;
      fieldPtr[2] = u_ref_[2] * u_max_;
    } else {
      fieldPtr[0] = u_ref_[0] * power_law_fn;
      fieldPtr[1] = u_ref_[1] * power_law_fn;
      fieldPtr[2] = u_ref_[2] * power_law_fn;
    }

    fieldPtr += fieldSize;
    coords += fieldSize;
  }
}

} // namespace nalu
} // namespace sierra
