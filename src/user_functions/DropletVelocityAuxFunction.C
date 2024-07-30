// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/DropletVelocityAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

DropletVelocityAuxFunction::DropletVelocityAuxFunction(
  const unsigned beginPos,
  const unsigned endPos,
  const std::vector<double>& params)
  : AuxFunction(beginPos, endPos),
    droppos_x_(0.0),
    droppos_y_(0.0),
    droppos_z_(0.0),
    dropvel_x_(0.1),
    dropvel_y_(0.1),
    dropvel_z_(0.1),
    radius_(0.1),
    interface_thickness_(0.0025)
{
  // check size and populate
  if (params.size() != 8 && !params.empty())
    throw std::runtime_error("Realm::setup_initial_conditions: "
                             "droplet (velocity) requires 8 params: 3 "
                             "components of droplet position, 3 "
                             "components of droplet velocity, droplet "
                             "radius, and interface thickness");
  if (!params.empty()) {
    droppos_x_ = params[0];
    droppos_y_ = params[1];
    droppos_z_ = params[2];
    dropvel_x_ = params[3];
    dropvel_y_ = params[4];
    dropvel_z_ = params[5];
    radius_ = params[6];
    interface_thickness_ = params[7];
  }
}

void
DropletVelocityAuxFunction::do_evaluate(
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

    auto rad_pos = std::sqrt(
                     (x - droppos_x_) * (x - droppos_x_) +
                     (y - droppos_y_) * (y - droppos_y_) +
                     (z - droppos_z_) * (z - droppos_z_)) -
                   radius_;
    auto vof = -0.5 * (std::erf(rad_pos / interface_thickness_) + 1.0) + 1.0;

    // Approximate average velocity by scaling with vof instead of using density
    fieldPtr[0] = vof * dropvel_x_;
    fieldPtr[1] = vof * dropvel_y_;
    fieldPtr[2] = vof * dropvel_z_;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
