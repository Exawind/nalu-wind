// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

#include <gtest/gtest.h>
#include "aero/actuator/UtilitiesActuator.h"
#include <aero/actuator/ActuatorTypes.h>
#include <cmath>
#include "NaluParsing.h"

namespace {

TEST(ActuatorFAST, NGP_unityGaussianTest)
{
  sierra::nalu::Coordinates epsilon;

  epsilon.x_ = std::pow(M_PI, -1.0 / 2.0);
  epsilon.y_ = std::pow(M_PI, -1.0 / 2.0);
  epsilon.z_ = std::pow(M_PI, -1.0 / 2.0);

  std::vector<double> epsilon_d{epsilon.x_, epsilon.y_, epsilon.z_};

  std::vector<double> distance{0, 0, 0};

  int nDim = 3;

  double r1 = sierra::nalu::actuator_utils::Gaussian_projection(
    nDim, distance.data(), epsilon);

  double r2 = sierra::nalu::actuator_utils::Gaussian_projection(
    nDim, distance.data(), epsilon_d.data());

  EXPECT_NEAR(1.0, r1, 1e-12);

  EXPECT_NEAR(r1, r2, 1e-12);
}

} // namespace
