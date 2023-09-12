// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/WaterLevelDensityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace sierra {
namespace nalu {

WaterLevelDensityAuxFunction::WaterLevelDensityAuxFunction(
  const std::vector<double>& params)
  : AuxFunction(0, 1),
    water_level_(0.),
    interface_thickness_(0.1),
    liquid_density_(1000.0),
    gas_density_(1.0)
{
  // check size and populate
  if (params.size() != 4 && !params.empty())
    throw std::runtime_error("Realm::setup_initial_conditions: "
                             "water_level requires 4 params: water level, "
                             "amplitude, kappa, and interface thickness");
  if (!params.empty()) {
    water_level_ = params[0];
    interface_thickness_ = params[1];
    liquid_density_ = params[2];
    gas_density_ = params[3];
  }
}

void
WaterLevelDensityAuxFunction::do_evaluate(
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

    const double z = coords[2];
    const double z0 = water_level_;

    const double local_vof =
      -0.5 * (std::erf((z - z0) / interface_thickness_) + 1.0) + 1.0;
    fieldPtr[0] = liquid_density_ * local_vof + gas_density_ * (1.0 - local_vof);

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
