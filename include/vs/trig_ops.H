#ifndef TRIG_OPS_H
#define TRIG_OPS_H

/** \file trig_ops.H
 *
 *  Trigonometric utilities
 */

#include <cmath>
#include "Kokkos_Core.hpp"

namespace utils {

//! Return \f$\pi\f$ as an double
KOKKOS_FORCEINLINE_FUNCTION constexpr double
pi()
{
  return M_PI;
}

//! Return \f$2 \pi\f$
KOKKOS_FORCEINLINE_FUNCTION constexpr double
two_pi()
{
  return 2.0 * M_PI;
}

//! Return \f$\pi / 2\f$
KOKKOS_FORCEINLINE_FUNCTION constexpr double
half_pi()
{
  return 0.5 * M_PI;
}

//! Convert from degrees to radians
KOKKOS_FORCEINLINE_FUNCTION double
radians(const double deg_val)
{
  return pi() * deg_val / 180.0;
}

//! Convert from radians to degrees
KOKKOS_FORCEINLINE_FUNCTION double
degrees(const double rad_val)
{
  return 180.0 * rad_val / pi();
}

} // namespace utils

#endif /* TRIG_OPS_H */
