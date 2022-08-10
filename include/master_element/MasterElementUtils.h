// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MasterElementUtils_h
#define MasterElementUtils_h

#include <array>
#include <limits>

#include <SimdInterface.h>
#include <KokkosInterface.h>

namespace sierra {
namespace nalu {

namespace MEconstants {
static const double realmin = std::numeric_limits<double>::min();
}
class LagrangeBasis;

bool isoparameteric_coordinates_for_point_2d(
  LagrangeBasis& basis,
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT pointCoord,
  double* POINTER_RESTRICT isoParCoord,
  std::array<double, 2> initialGuess,
  int maxIter,
  double tolerance,
  double deltaLimit = 1.0e4);

bool isoparameteric_coordinates_for_point_3d(
  LagrangeBasis& basis,
  const double* POINTER_RESTRICT elemNodalCoords,
  const double* POINTER_RESTRICT pointCoord,
  double* POINTER_RESTRICT isoParCoord,
  std::array<double, 3> initialGuess,
  int maxIter,
  double tolerance,
  double deltaLimit = 1.0e4);

} // namespace nalu
} // namespace sierra

#endif
