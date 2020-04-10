#include "matrix_free/GaussLegendreQuadratureRule.h"

#include "Kokkos_Array.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
constexpr Kokkos::Array<double, 1> LGL<1>::nodes;
constexpr Kokkos::Array<double, 2> LGL<2>::nodes;
constexpr Kokkos::Array<double, 3> LGL<3>::nodes;
constexpr Kokkos::Array<double, 5> LGL<5>::nodes;
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
