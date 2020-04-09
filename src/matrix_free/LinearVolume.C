#include "matrix_free/LinearVolume.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/TensorOperations.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

#include <Kokkos_Macros.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {

namespace {

#define LN 0
#define RN 1
#define XH 0
#define YH 1
#define ZH 2

template <int p, int dj, int di, typename CoeffArray, typename BoxArray>
KOKKOS_FUNCTION typename BoxArray::value_type
hex_jacobian_component(
  const CoeffArray& Nlin, const BoxArray& box, int k, int j, int i)
{
  if (dj == XH) {
    return (-Nlin(LN, j) * Nlin(LN, k) * box(di, 0) +
            Nlin(LN, j) * Nlin(LN, k) * box(di, 1) +
            Nlin(RN, j) * Nlin(LN, k) * box(di, 2) -
            Nlin(RN, j) * Nlin(LN, k) * box(di, 3) -
            Nlin(LN, j) * Nlin(RN, k) * box(di, 4) +
            Nlin(LN, j) * Nlin(RN, k) * box(di, 5) +
            Nlin(RN, j) * Nlin(RN, k) * box(di, 6) -
            Nlin(RN, j) * Nlin(RN, k) * box(di, 7)) *
           0.5;
  } else if (dj == YH) {
    return (-Nlin(LN, i) * Nlin(LN, k) * box(di, 0) -
            Nlin(RN, i) * Nlin(LN, k) * box(di, 1) +
            Nlin(RN, i) * Nlin(LN, k) * box(di, 2) +
            Nlin(LN, i) * Nlin(LN, k) * box(di, 3) -
            Nlin(LN, i) * Nlin(RN, k) * box(di, 4) -
            Nlin(RN, i) * Nlin(RN, k) * box(di, 5) +
            Nlin(RN, i) * Nlin(RN, k) * box(di, 6) +
            Nlin(LN, i) * Nlin(RN, k) * box(di, 7)) *
           0.5;
  } else {
    return (-Nlin(LN, i) * Nlin(LN, j) * box(di, 0) -
            Nlin(RN, i) * Nlin(LN, j) * box(di, 1) -
            Nlin(RN, i) * Nlin(RN, j) * box(di, 2) -
            Nlin(LN, i) * Nlin(RN, j) * box(di, 3) +
            Nlin(LN, i) * Nlin(LN, j) * box(di, 4) +
            Nlin(RN, i) * Nlin(LN, j) * box(di, 5) +
            Nlin(RN, i) * Nlin(RN, j) * box(di, 6) +
            Nlin(LN, i) * Nlin(RN, j) * box(di, 7)) *
           0.5;
  }
}

template <int p, typename CoeffArray, typename BoxArray>
KOKKOS_FUNCTION LocalArray<typename BoxArray::value_type[3][3]>
linear_hex_jacobian(
  const CoeffArray& coeff, const BoxArray& box, int k, int j, int i)
{
  LocalArray<typename BoxArray::value_type[3][3]> jac;
  jac(0, 0) = hex_jacobian_component<p, XH, XH>(coeff, box, k, j, i);
  jac(0, 1) = hex_jacobian_component<p, XH, YH>(coeff, box, k, j, i);
  jac(0, 2) = hex_jacobian_component<p, XH, ZH>(coeff, box, k, j, i);
  jac(1, 0) = hex_jacobian_component<p, YH, XH>(coeff, box, k, j, i);
  jac(1, 1) = hex_jacobian_component<p, YH, YH>(coeff, box, k, j, i);
  jac(1, 2) = hex_jacobian_component<p, YH, ZH>(coeff, box, k, j, i);
  jac(2, 0) = hex_jacobian_component<p, ZH, XH>(coeff, box, k, j, i);
  jac(2, 1) = hex_jacobian_component<p, ZH, YH>(coeff, box, k, j, i);
  jac(2, 2) = hex_jacobian_component<p, ZH, ZH>(coeff, box, k, j, i);
  return jac;
}

#undef LN
#undef RN
#undef XH
#undef YH
#undef ZH

} // namespace

template <int p>
scalar_view<p>
volume_metric_t<p>::invoke(
  const const_scalar_view<p> alpha, const const_vector_view<p> coordinates)
{
  constexpr auto nlin = Coeffs<p>::Nlin;
  constexpr int n1D = p + 1;
  scalar_view<p> volume("volumes", coordinates.extent_int(0));
  Kokkos::parallel_for(
    "volume", coordinates.extent_int(0), KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int k = 0; k < n1D; ++k) {
        for (int j = 0; j < n1D; ++j) {
          for (int i = 0; i < n1D; ++i) {
            volume(index, k, j, i) =
              alpha(index, k, j, i) *
              determinant<ftype>(linear_hex_jacobian<p>(nlin, box, k, j, i));
          }
        }
      }
    });
  return volume;
}
INSTANTIATE_POLYSTRUCT(volume_metric_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
