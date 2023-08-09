// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <user_functions/SloshingTankVOFAuxFunction.h>
#include <algorithm>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra {
namespace nalu {

SloshingTankVOFAuxFunction::SloshingTankVOFAuxFunction() : AuxFunction(0, 1)
{
  // does nothing
}

void
SloshingTankVOFAuxFunction::do_evaluate(
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
    const double interface_thickness = 0.025;

    const double water_level = 0.;
    const double Amp = 0.1;
    const double kappa = 0.25;
    const double Lx = 20.;
    const double Ly = 20.;

    const double z0 =
                    water_level +
                    Amp * std::exp(
                              -kappa * (std::pow(x - 0.5 * Lx, 2) +
                                        std::pow(y - 0.5 * Ly, 2)));
                fieldPtr[0] = z0 - z;

    auto radius = std::sqrt(x * x + y * y + z * z) - 0.075;
    fieldPtr[0] += -0.5 * (std::erf(radius / interface_thickness) + 1.0) + 1.0;

    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace sierra
