// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.

namespace sierra {
namespace nalu {

struct Coordinates;

namespace actuator_utils {

// A Gaussian projection function
double Gaussian_projection(
  const int &nDim,
  double *dis,
  const Coordinates &epsilon);

// A Gaussian projection function
double Gaussian_projection(
  const int &nDim,
  double *dis,
  double *epsilon);

}  // namespace actuator_utils
}  // namespace actuator_utils
}  // namespace actuator_utils
