#include "matrix_free/LinearAreas.h"

#include "Kokkos_DualView.hpp"
#include "Kokkos_Macros.hpp"

#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace geom {
namespace impl {
namespace {

template <int p, int dk, int dj, int di, typename BoxArray>
KOKKOS_FUNCTION typename BoxArray::value_type
hex_jacobian_component_scs(const BoxArray& box, int l, int s, int r)
{
  enum { LN = 0, RN = 1 };
  enum { XH = 0, YH = 1, ZH = 2 };

  static constexpr auto nlin = Coeffs<p>::Nlin;
  static constexpr auto ntlin = Coeffs<p>::Ntlin;
  typename BoxArray::value_type jac(0);
  switch (dj) {
  case XH: {
    const double lj =
      (dk == YH) ? ntlin(LN, l) : (dk == XH) ? nlin(LN, r) : nlin(LN, s);
    const double rj =
      (dk == YH) ? ntlin(RN, l) : (dk == XH) ? nlin(RN, r) : nlin(RN, s);

    const double lk = (dk == ZH) ? ntlin(LN, l) : nlin(LN, s);
    const double rk = (dk == ZH) ? ntlin(RN, l) : nlin(RN, s);

    jac = -lj * lk * box(di, 0) + lj * lk * box(di, 1) + rj * lk * box(di, 2) -
          rj * lk * box(di, 3) - lj * rk * box(di, 4) + lj * rk * box(di, 5) +
          rj * rk * box(di, 6) - rj * rk * box(di, 7);
    break;
  }
  case YH: {
    const double li = (dk == XH) ? ntlin(LN, l) : nlin(LN, r);
    const double ri = (dk == XH) ? ntlin(RN, l) : nlin(RN, r);

    const double lk = (dk == ZH) ? ntlin(LN, l) : nlin(LN, s);
    const double rk = (dk == ZH) ? ntlin(RN, l) : nlin(RN, s);

    jac = -li * lk * box(di, 0) - ri * lk * box(di, 1) + ri * lk * box(di, 2) +
          li * lk * box(di, 3) - li * rk * box(di, 4) - ri * rk * box(di, 5) +
          ri * rk * box(di, 6) + li * rk * box(di, 7);
    break;
  }
  case ZH: {
    const double li = (dk == XH) ? ntlin(LN, l) : nlin(LN, r);
    const double ri = (dk == XH) ? ntlin(RN, l) : nlin(RN, r);

    const double lj =
      (dk == YH) ? ntlin(LN, l) : (dk == XH) ? nlin(LN, r) : nlin(LN, s);
    const double rj =
      (dk == YH) ? ntlin(RN, l) : (dk == XH) ? nlin(RN, r) : nlin(RN, s);

    jac = -li * lj * box(di, 0) - ri * lj * box(di, 1) - ri * rj * box(di, 2) -
          li * rj * box(di, 3) + li * lj * box(di, 4) + ri * lj * box(di, 5) +
          ri * rj * box(di, 6) + li * rj * box(di, 7);
    break;
  }
  default:
    break;
  }
  constexpr double isoParametricFactor = 0.5;
  return jac * isoParametricFactor;
}

template <int p, int dk, typename BoxArray>
KOKKOS_FUNCTION LocalArray<ftype[3]>
linear_area(const BoxArray& box, int k, int j, int i)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  static constexpr int ds1 = (dk == XH) ? ZH : (dk == YH) ? XH : YH;
  static constexpr int ds2 = (dk == XH) ? YH : (dk == YH) ? ZH : XH;
  const auto dx_ds1 = hex_jacobian_component_scs<p, dk, ds1, XH>(box, k, j, i);
  const auto dx_ds2 = hex_jacobian_component_scs<p, dk, ds2, XH>(box, k, j, i);
  const auto dy_ds1 = hex_jacobian_component_scs<p, dk, ds1, YH>(box, k, j, i);
  const auto dy_ds2 = hex_jacobian_component_scs<p, dk, ds2, YH>(box, k, j, i);
  const auto dz_ds1 = hex_jacobian_component_scs<p, dk, ds1, ZH>(box, k, j, i);
  const auto dz_ds2 = hex_jacobian_component_scs<p, dk, ds2, ZH>(box, k, j, i);
  return LocalArray<ftype[3]>{
    {dy_ds1 * dz_ds2 - dz_ds1 * dy_ds2, dz_ds1 * dx_ds2 - dx_ds1 * dz_ds2,
     dx_ds1 * dy_ds2 - dy_ds1 * dx_ds2}};
}

} // namespace

template <int p>
scs_vector_view<p>
linear_areas_t<p>::invoke(const_vector_view<p> coordinates)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  scs_vector_view<p> area("area", coordinates.extent_int(0));
  Kokkos::parallel_for(
    coordinates.extent_int(0), KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, coordinates);
      for (int l = 0; l < p; ++l) {
        for (int s = 0; s < p + 1; ++s) {
          for (int r = 0; r < p + 1; ++r) {
            auto av = linear_area<p, XH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              area(index, XH, l, s, r, d) = av(d);
            }

            av = linear_area<p, YH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              area(index, YH, l, s, r, d) = av(d);
            }

            av = linear_area<p, ZH>(box, l, s, r);
            for (int d = 0; d < 3; ++d) {
              area(index, ZH, l, s, r, d) = av(d);
            }
          }
        }
      }
    });
  return area;
}

INSTANTIATE_POLYSTRUCT(linear_areas_t);

} // namespace impl
} // namespace geom
} // namespace matrix_free
} // namespace nalu
} // namespace sierra