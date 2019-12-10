/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
/*
 * UnitTestActuatorDiskFAST.C
 *
 *  Created on: Nov, 2019
 *      Author: tony
 */

#include <gtest/gtest.h>
#include "actuator/UtilitiesActuator.h"
#include <cmath>
#include "NaluParsing.h"

namespace {

TEST(ActuatorFAST, gaussianTest)
  {

  //  
  // This test computes a Gaussian value of unity
  //  

  // Establish the value of epsilon
  sierra::nalu::Coordinates epsilon;

  // Populate the values
  epsilon.x_ = std::pow(M_PI, -1.0/2.0);
  epsilon.y_ = std::pow(M_PI, -1.0/2.0);
  epsilon.z_ = std::pow(M_PI, -1.0/2.0);

  // Create a new epsilon of type double
  std::vector<double> epsilon_d{epsilon.x_, epsilon.y_, epsilon.z_};

  // Create a distance vector
  std::vector<double> distance{0, 0, 0};

  // The number of dimensions
  int nDim = 3;

  // Call the Gaussian smearing function
  double r1 = sierra::nalu::actuator_utils::Gaussian_projection(
    nDim, distance.data(), epsilon);

  // Call the Gaussian smearing function
  double r2 = sierra::nalu::actuator_utils::Gaussian_projection(
    nDim, distance.data(), epsilon_d.data());

  // Check that the math of the Gaussian is computed accordingly
  //expected value, computed value, tolerance
  EXPECT_NEAR(1.0, r1, 1e-12);

  // Check that both functions produce same result
  //expected value, computed value, tolerance
  EXPECT_NEAR(r1, r2, 1e-12);

  }

}  // namespace
