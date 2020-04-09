#ifndef ELEMENT_VOLUME_INTEGRAL_H
#define ELEMENT_VOLUME_INTEGRAL_H

#include <cmath>

#include "matrix_free/Coefficients.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"
#include "matrix_free/ShuffledAccess.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p, int dir, typename InArray, typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
edge_integral(const InArray& in, OutArray& out)
{
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        const int l = impl::active_index<dir>::index_0(k, j, i);
        const int m = impl::active_index<dir>::index_1(k, j, i);
        const int n = impl::active_index<dir>::index_2(k, j, i);
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          static constexpr auto vandermonde = Coeffs<p>::W;
          acc += vandermonde(l, q) * shuffled_access<dir>(in, n, m, q);
        }
        out(k, j, i) = acc;
      }
    }
  }
}

template <int p, typename InArray, typename ScratchArray, typename OutArray>
KOKKOS_FUNCTION void
volume(const InArray& in, ScratchArray& scratch, OutArray& out)
{
  edge_integral<p, 0>(in, out);
  edge_integral<p, 1>(out, scratch);
  edge_integral<p, 2>(scratch, out);
}

template <
  int p,
  typename VolumeArray,
  typename InArray,
  typename ScratchArray,
  typename OutArray>
KOKKOS_FUNCTION void
consistent_mass_time_derivative(
  int index,
  Kokkos::Array<double, 3> gammas,
  const VolumeArray& vol,
  const InArray& qm1,
  const InArray& qp0,
  const InArray& qp1,
  ScratchArray& scratch,
  OutArray& out)
{
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        scratch(k, j, i) =
          -vol(index, k, j, i) *
          (gammas[0] * qp1(index, k, j, i) + gammas[1] * qp0(index, k, j, i) +
           gammas[2] * qm1(index, k, j, i));
      }
    }
  }
  edge_integral<p, 0>(scratch, out);
  edge_integral<p, 1>(out, scratch);
  edge_integral<p, 2>(scratch, out);
}

template <int p, typename VolumeArray, typename InArray, typename OutArray>
KOKKOS_FUNCTION void
lumped_time_derivative(
  int index,
  Kokkos::Array<double, 3> gammas,
  const VolumeArray& vol,
  const InArray& qm1,
  const InArray& qp0,
  const InArray& qp1,
  OutArray& out)
{
  static constexpr auto lumped = Coeffs<p>::Wl;
  for (int k = 0; k < p + 1; ++k) {
    const auto Wk = lumped[k];
    for (int j = 0; j < p + 1; ++j) {
      const auto WkWj = Wk * lumped[j];
      for (int i = 0; i < p + 1; ++i) {
        out(k, j, i) =
          -WkWj * lumped[i] * vol(index, k, j, i) *
          (gammas[0] * qp1(index, k, j, i) + gammas[1] * qp0(index, k, j, i) +
           gammas[2] * qm1(index, k, j, i));
      }
    }
  }
}

template <
  int p,
  typename VolumeArray,
  typename DeltaArray,
  typename ScratchArray,
  typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
mass_term(
  int index,
  double gamma,
  const VolumeArray& vol,
  const DeltaArray& delta,
  ScratchArray& scratch,
  OutArray& out)
{
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        scratch(k, j, i) = -gamma * vol(index, k, j, i) * delta(k, j, i);
      }
    }
  }
  edge_integral<p, 0>(scratch, out);
  edge_integral<p, 1>(out, scratch);
  edge_integral<p, 2>(scratch, out);
}

template <int p, typename VolumeArray, typename DeltaArray, typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
lumped_mass_term(
  int index,
  double gamma,
  const VolumeArray& vol,
  const DeltaArray& delta,
  OutArray& out)
{
  static constexpr auto lumped = Coeffs<p>::Wl;

  for (int k = 0; k < p + 1; ++k) {
    const auto gammaWk = -gamma * lumped(k);
    for (int j = 0; j < p + 1; ++j) {
      const auto gammaWkWj = lumped(j) * gammaWk;
      for (int i = 0; i < p + 1; ++i) {
        out(k, j, i) =
          gammaWkWj * lumped(i) * vol(index, k, j, i) * delta(k, j, i);
      }
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
