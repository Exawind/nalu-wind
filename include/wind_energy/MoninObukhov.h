/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MONINOBUKHOV_H
#define MONINOBUKHOV_H

#include "KokkosInterface.h"
#include "SimdInterface.h"

namespace sierra {
namespace nalu {
namespace abl_monin_obukhov {

/**
 * van der Laan, P., Kelly, M. C., & Sørensen, N. N. (2017). A new k-epsilon
 * model consistent with Monin-Obukhov similarity theory. Wind Energy, 20(3),
 * 479–489. https://doi.org/10.1002/we.2017
 */

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
T psim_stable(const T& zeta, const T beta = 5.0)
{
  return (-beta * zeta);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
T psih_stable(const T& zeta, const T beta = 5.0)
{
  return (-beta * zeta);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
T psim_unstable(const T& zeta, const T gamma = 16.0)
{
  static constexpr double half_pi = 1.5707963267948966;

  // Actually (1.0 / phim)
  const T phim = stk::math::pow((1.0 - gamma * zeta), 0.25);
  const T term1 = (1.0 + phim * phim);
  const T term2 = (1.0 + phim);
  const T term3  = term2 * term2;
  const T term4 = stk::math::log(0.125 * term1 * term3);
  const T term5 = 2.0 * stk::math::atan(phim);

  return (term4 - term5 + half_pi);
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
T psih_unstable(const T& zeta, const T gamma = 16.0)
{
  // Actually (1.0 / phih)
  const T phih = stk::math::sqrt(1.0 - gamma * zeta);
  return (stk::math::log(0.5 * (1.0 + phih)));
}

}
}  // nalu
}  // sierra


#endif /* MONINOBUKHOV_H */
