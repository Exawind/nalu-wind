// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef TAMSUTILS_H
#define TAMSUTILS_H

#include <SimdInterface.h>

namespace sierra {
namespace nalu {

namespace tams_utils {

template <class T, int dim = 3>
KOKKOS_FUNCTION
T
get_M43_constant(T D[dim][dim], const double CMdeg)
{
  NGP_ThrowRequireMsg(dim == 3, "Compute of M43 constant requires 3D problem");

  // Coefficients for the polynomial
  double c[15] = {1.033749474513071,  -0.154122686264488, -0.007737595743644,
                  0.177611732560139,  0.060868024017604,  0.162200630336440,
                  -0.041086757724764, -0.027380130027626, 0.005521188430182,
                  0.049139605169403,  0.002926283060215,  0.002672790587853,
                  0.000486437925728,  0.002136258066662,  0.005113058518679};

  T smallestEV = stk::math::min(D[0][0], stk::math::min(D[1][1], D[2][2]));
  T largestEV = stk::math::max(D[0][0], stk::math::max(D[1][1], D[2][2]));
  T middleEV = stk::math::if_then_else(
    D[0][0] == smallestEV, stk::math::min(D[1][1], D[2][2]),
    stk::math::if_then_else(
      D[1][1] == smallestEV, stk::math::min(D[0][0], D[2][2]),
      stk::math::min(D[0][0], D[1][1])));

  // Scale the EVs
  middleEV = middleEV / smallestEV;
  largestEV = largestEV / smallestEV;

  T r =
    stk::math::sqrt(stk::math::pow(middleEV, 2) + stk::math::pow(largestEV, 2));
  T theta = stk::math::acos(largestEV / r);

  T x = stk::math::log(r);
  T y = stk::math::log(stk::math::sin(2 * theta));

  T poly = c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y +
           c[5] * y * y + c[6] * x * x * x + c[7] * x * x * y +
           c[8] * x * y * y + c[9] * y * y * y + c[10] * x * x * x * x +
           c[11] * x * x * x * y + c[12] * x * x * y * y +
           c[13] * x * y * y * y + c[14] * y * y * y * y;

  return poly * CMdeg;
}

} // namespace tams_utils

} // namespace nalu
} // namespace sierra

#endif /* TAMSUTILS_H */
