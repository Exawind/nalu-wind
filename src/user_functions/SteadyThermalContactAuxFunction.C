// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include <user_functions/SteadyThermalContactAuxFunction.h>
#include <algorithm>
#include <stk_mesh/base/MetaData.hpp>
#include "utils/StkHelpers.h"

#include <yaml-cpp/yaml.h>

// basic c++
#include <cmath>
#include <vector>
#include <stdexcept>

namespace sierra{
namespace nalu{

void execute(const SteadyThermalContactData& data, const stk::mesh::FastMeshIndex& mi)
{
  const double x = data.coordinate_field.get(mi, 0);
  const double y = data.coordinate_field.get(mi, 1);
  data.temperature_field.get(mi, 0) = data.amplitude *
      (std::cos(2 * M_PI * data.wave_number * x) + std::cos(2 * M_PI * data.wave_number * y));
}

SteadyThermalContactAuxFunction::SteadyThermalContactAuxFunction() :
  AuxFunction(0,1),
  a_(1.0),
  k_(1.0),
  pi_(std::acos(-1.0))
{
  // nothing to do
}


void
SteadyThermalContactAuxFunction::do_evaluate(
  const double *coords,
  const double /* t */,
  const unsigned spatialDimension,
  const unsigned numPoints,
  double * fieldPtr,
  const unsigned fieldSize,
  const unsigned /*beginPos*/,
  const unsigned /*endPos*/) const
{
  for(unsigned p=0; p < numPoints; ++p) {

    double x = coords[0];
    double y = coords[1];

    fieldPtr[0] = k_/4.0*(cos(2.*a_*pi_*x) + cos(2.*a_*pi_*y));
    
    fieldPtr += fieldSize;
    coords += spatialDimension;
  }
}

} // namespace nalu
} // namespace Sierra
