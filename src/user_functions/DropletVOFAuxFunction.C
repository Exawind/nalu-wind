// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/DropletVOFAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

DropletVOFAuxFunction::DropletVOFAuxFunction(const std::vector<double>& params)
  : AuxFunction(0, 1),
    droppos_x_(0.0),
    droppos_y_(0.0),
    droppos_z_(0.0),
    radius_(0.1),
    interface_thickness_(0.0025)
{
  // check size and populate
  if (params.size() != 5 && !params.empty())
    throw std::runtime_error("Realm::setup_initial_conditions: "
                             "droplet (volume_of_fluid) requires 5 params: 3 "
                             "components of droplet position, droplet "
                             "radius, and interface thickness");
  if (!params.empty()) {
    droppos_x_ = params[0];
    droppos_y_ = params[1];
    droppos_z_ = params[2];
    radius_ = params[3];
    interface_thickness_ = params[4];
  }
}

void
DropletVOFAuxFunction::do_evaluate(
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
    const double z = coords[2];

    // fieldPtr[0] = 0.0;
    // fieldPtr[0] += -0.5 * (std::erf(y / interface_thickness) + 1.0) + 1.0;

    auto rad_pos =
      std::sqrt(
        (x - droppos_x_) * (x - droppos_x_) + (y - droppos_y_) * (y - droppos_y_) +
        (z - droppos_z_) * (z - droppos_z_)) -
      radius_;
    // fieldPtr[0] += -0.5 * (std::erf(radius / interface_thickness) + 1.0)
    // + 1.0;
    fieldPtr[0] = -0.5 * (std::erf(rad_pos / interface_thickness_) + 1.0) + 1.0;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
