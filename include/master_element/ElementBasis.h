// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ELEMENT_BASIS_H
#define ELEMENT_BASIS_H

#include "AlgTraits.h"
#include "ArrayND.h"

namespace sierra::nalu {

template <typename ViewLikeT>
using val_t = typename ViewLikeT::value_type;

struct Tri3Basis
{
  using traits_t = AlgTraitsTri3_2D;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  template <typename LocT>
  static constexpr double interpolant(int n, const LocT& x)
  {
    return (n > 0) ? x[n - 1] : 1 - (x[0] + x[1]);
  }

  template <typename LocT>
  static constexpr auto deriv_coeff(int n, const LocT& /*x*/, int d)
  {
    return -1. + (n > 0) * (1. + (d == (n - 1)));
  }
};

struct LineBasis
{
  using traits_t = AlgTraitsEdge_2D;
  static constexpr int N1D = 2;
  static constexpr int DIM = 1;
  static constexpr int NNODES = N1D;

  template <typename LocT>
  static constexpr double interpolant(int n, const LocT& x)
  {
    return 0.5 * (1 + (2 * n - 1) * x[0]);
  }

  template <typename LocT>
  static constexpr auto deriv_coeff(int n, const LocT& /*x*/, int /*d*/)
  {
    return 0.5 * (2 * n - 1);
  }
};

struct Quad42DBasis
{
  using traits_t = AlgTraitsQuad4_2D;
  static constexpr int N1D = 2;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  [[nodiscard]] static constexpr ArrayND<int[2]> to_tensor(int n)
  {
    constexpr ArrayND<int[NNODES][2]> map{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    return {map(n, 0), map(n, 1)};
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto interp_1_2D(int l, const LocT& x)
  {
    return 0.5 * (1 + (2 * l - 1) * x);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_1_2D(int l, const LocT& /*unused*/)
  {
    return 0.5 * (2 * l - 1);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr double interpolant(int n, const LocT& x)
  {
    const auto ij = to_tensor(n);
    return interp_1_2D(ij[0], x[0]) * interp_1_2D(ij[1], x[1]);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_coeff(int n, const LocT& x, int d)
  {
    // for consistency in metric tensor computations, rescale back to-1/2,1/2
    // range
    constexpr int iso_range = 2;

    const auto ij = to_tensor(n);
    if (d == 0) {
      return iso_range * deriv_1_2D(ij(0), x[0]) * interp_1_2D(ij(1), x[1]);
    }
    return iso_range * interp_1_2D(ij(0), x[0]) * deriv_1_2D(ij(1), x[1]);
  }
};

struct Quad4Basis
{
  using traits_t = AlgTraitsQuad4_2D;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  template <typename LocT>
  [[nodiscard]] static constexpr double interpolant(int n, const LocT& x)
  {
    return (n > 0) ? x[n - 1] : 1 - (x[0] + x[1] + x[2]);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto
  deriv_coeff(int n, const LocT& /*x*/, int d)
  {
    return -1. + (n > 0) * (1. + (d == (n - 1)));
  }
};

struct Tet4Basis
{
  using traits_t = AlgTraitsTet4;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  template <typename LocT>
  [[nodiscard]] static constexpr double interpolant(int n, const LocT& x)
  {
    return (n > 0) ? x[n - 1] : 1 - (x[0] + x[1] + x[2]);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto
  deriv_coeff(int n, const LocT& /*x*/, int d)
  {
    return -1. + (n > 0) * (1. + (d == (n - 1)));
  }
};

struct Pyr5Basis
{
  using traits_t = AlgTraitsPyr5;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  [[nodiscard]] static constexpr int sgn(int n, int d)
  {
    constexpr auto map =
      ArrayND<int[4][2]>{{{-1, -1}, {+1, -1}, {+1, +1}, {-1, +1}}};
    return map(n % 4, d);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr double interpolant(int n, const LocT& x)
  {
    if (n >= 4 || x[2] == val_t<LocT>(1.)) {
      return x[2];
    }
    return 0.25 * (1 + sgn(n, 0) * x[0] - x[2]) *
           (1 + sgn(n, 1) * x[1] - x[2]) / (1 - x[2]);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_coeff(int n, const LocT& x, int d)
  {
    if (n >= 4 || x[2] == 1.) {
      return val_t<LocT>(d == 2);
    }
    const auto apex = 0.25 / (1 - x[2]);
    const auto square_x = 1 + sgn(n, 0) * x[0] - x[2];
    const auto square_y = 1 + sgn(n, 1) * x[1] - x[2];
    const ArrayND<val_t<LocT>[3]> dsquare{
      {sgn(n, 0) * square_y, square_x * sgn(n, 1), -(square_x + square_y)}};
    const ArrayND<val_t<LocT>[3]> dinv_term{0, 0, 4 * apex * apex};
    return dsquare(d) * apex + square_x * square_y * dinv_term(d);
  }
};

struct Pyr5DegenHexBasis
{
  using traits_t = AlgTraitsPyr5;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  [[nodiscard]] static constexpr int sgn(int n, int d)
  {
    constexpr auto map =
      ArrayND<int[4][2]>{{{-1, -1}, {+1, -1}, {+1, +1}, {-1, +1}}};
    return map(n % 4, d);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto interpolant(int n, const LocT& x)
  {
    if (n >= 4) {
      return x[2];
    }
    return 0.25 * (1 + sgn(n, 0) * x[0]) * (1 + sgn(n, 1) * x[1]) * (1 - x[2]);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_coeff(int n, const LocT& x, int d)
  {
    if (n >= 4) {
      return val_t<LocT>(d == 2);
    }
    const ArrayND<val_t<LocT>[3]> dsq = {
      sgn(n, 0) * (1 + sgn(n, 1) * x[1]), (1 + sgn(n, 0) * x[0]) * sgn(n, 1),
      0};
    return 0.25 * dsq(d) * (1 - x[2]) -
           (d == 2) * (1 + sgn(n, 0) * x[0]) * (1 + sgn(n, 1) * x[1]);
  }
};

struct Wed6Basis
{
  using traits_t = AlgTraitsWed6;
  static constexpr int DIM = traits_t::nDim_;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  template <typename LocT>
  [[nodiscard]] static constexpr double interpolant(int n, const LocT& x)
  {
    const ArrayND<typename LocT::value_type[3]> tri{
      1 - (x[0] + x[1]), x[0], x[1]};
    return tri(n % 3) * (0.5 * (1 + (2 * (n > 2) - 1) * x[2]));
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_coeff(int n, const LocT& x, int d)
  {
    const ArrayND<val_t<LocT>[3]> tri{1 - (x[0] + x[1]), x[0], x[1]};
    constexpr ArrayND<val_t<LocT>[3][3]> dtri{
      {{-1, -1, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto prism = 0.5 * (1 + (2 * (n > 2) - 1) * x[2]);
    const auto dprism = ArrayND<val_t<LocT>[3]>{0, 0, 0.5 * (2 * (n > 2) - 1)};
    return dtri(n % 3, d) * prism + tri(n % 3) * dprism(d);
  }
};

struct Hex8Basis
{
  using traits_t = AlgTraitsHex8;
  static constexpr int N1D = 2;
  static constexpr int DIM = 3;
  static constexpr int NNODES = traits_t::nodesPerElement_;

  [[nodiscard]] static constexpr int tensor_map(int k, int j, int i)
  {
    constexpr ArrayND<int[N1D][N1D][N1D]> map{
      {{{0, 1}, {3, 2}}, {{4, 5}, {7, 6}}}};
    return map(k, j, i);
  }

  [[nodiscard]] static constexpr ArrayND<int[3]> to_tensor(int n)
  {
    constexpr ArrayND<int[NNODES][3]> map{
      {{0, 0, 0},
       {1, 0, 0},
       {1, 1, 0},
       {0, 1, 0},
       {0, 0, 1},
       {1, 0, 1},
       {1, 1, 1},
       {0, 1, 1}}};
    return {map(n, 0), map(n, 1), map(n, 2)};
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto interp_1(int l, const LocT& x)
  {
    return 0.5 * (1 + (2 * l - 1) * x);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_1(int l, const LocT& /*unused*/)
  {
    return 0.5 * (2 * l - 1);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr double interpolant(int n, const LocT& x)
  {
    const auto ijk = to_tensor(n);
    return interp_1(ijk[0], x[0]) * interp_1(ijk[1], x[1]) *
           interp_1(ijk[2], x[2]);
  }

  template <typename LocT>
  [[nodiscard]] static constexpr auto deriv_coeff(int n, const LocT& x, int d)
  {
    constexpr auto iso_range =
      2; // for historical reasons, rescale to -1/2, 1/2

    const auto ijk = to_tensor(n);
    const auto xv = (d == 0) ? deriv_1(ijk(0), x[0]) : interp_1(ijk(0), x[0]);
    const auto yv = (d == 1) ? deriv_1(ijk(1), x[1]) : interp_1(ijk(1), x[1]);
    const auto zv = (d == 2) ? deriv_1(ijk(2), x[2]) : interp_1(ijk(2), x[2]);
    return iso_range * xv * yv * zv;
  }
};

namespace impl {
template <typename AlgTraits>
struct BasisSelector
{
};
template <>
struct BasisSelector<AlgTraitsTet4>
{
  using basis_t = Tet4Basis;
};
template <>
struct BasisSelector<AlgTraitsPyr5>
{
  using basis_t = Pyr5Basis;
};
template <>
struct BasisSelector<AlgTraitsWed6>
{
  using basis_t = Wed6Basis;
};
template <>
struct BasisSelector<AlgTraitsHex8>
{
  using basis_t = Hex8Basis;
};
} // namespace impl

template <typename AlgTraits>
using alg_basis_t = typename impl::BasisSelector<AlgTraits>::basis_t;

} // namespace sierra::nalu

namespace sierra::nalu::utils {

template <class T, class = void>
struct is_tensor_compatible : std::false_type
{
};

template <class T>
struct is_tensor_compatible<T, std::void_t<decltype(&T::interp_1)>>
  : std::true_type
{
};

template <typename BasisT, typename ParCoordsArrayT, typename ValArrayT>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
  is_tensor_compatible<BasisT>::value,
  ArrayND<typename ValArrayT::value_type[ParCoordsArrayT::extent_int(0)]
                                        [ValArrayT::extent_int(1)]>>
interpolate(const ParCoordsArrayT& par_coords)
{
  constexpr int nparcoords = ParCoordsArrayT::extent_int(0);
  using res_val_t = typename ValArrayT::value_type;
  using array_t = res_val_t[nparcoords];

  ArrayND<array_t> result{};
  for (int l = 0; l < nparcoords; ++l) {
    for (int d = 0; d < ValArrayT::extent_int(1); ++d) {
      result(l, d) = 0;
    }
    for (int i = 0; i < BasisT::N1D; ++i) {
      const auto interp_x = BasisT::interp_1(i, par_coords(l, 0));
      for (int j = 0; j < BasisT::N1D; ++j) {
        const auto interp_xy = BasisT::interp_1(j, par_coords(l, 1)) * interp_x;
        for (int k = 0; k < BasisT::N1D; ++k) {
          const auto interp_xyz =
            interp_xy * BasisT::interp_1(i, par_coords(l, 2));
          for (int d = 0; d < ValArrayT::extent_int(1); ++d) {
            result(l, d) += interp_xyz * vals(BasisT::tensor_map(k, j, i), d);
          }
        }
      }
    }
  }
  return result;
}

template <typename BasisT, typename ParCoordsArrayT, typename ValArrayT>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::enable_if_t<
  !is_tensor_compatible<BasisT>::value,
  ArrayND<typename ValArrayT::value_type[ParCoordsArrayT::extent_int(0)]
                                        [ValArrayT::extent_int(1)]>>
interpolate(const ParCoordsArrayT& par_coords, const ValArrayT& vals)
{
  constexpr int nparcoords = ParCoordsArrayT::extent_int(0);
  constexpr int nvals = ValArrayT::extent_int(1);
  using res_val_t = std::decay_t<decltype(par_coords(0, 0) * vals(0, 0))>;
  using array_t = res_val_t[nparcoords][nvals];

  ArrayND<array_t> result{};
  for (int l = 0; l < nparcoords; ++l) {

    ArrayND<res_val_t[ParCoordsArrayT::extent_int(1)]> point;
    for (int d = 0; d < ParCoordsArrayT::extent_int(1); ++d) {
      point(d) = par_coords(l, d);
    }
    for (int d = 0; d < ValArrayT::extent_int(1); ++d) {
      result(l, d) = 0;
    }

    for (int n = 0; n < BasisT::NNODES; ++n) {
      const auto interp_n = BasisT::interp(n, point);
      for (int d = 0; d < ValArrayT::extent_int(1); ++d) {
        result(l, d) += interp_n * vals(n, d);
      }
    }
  }
  return result;
}

template <typename BasisT, typename ParCoordsArrayT>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr ArrayND<
  typename ParCoordsArrayT::value_type[ParCoordsArrayT::extent_int(0)]
                                      [BasisT::NNODES]>
interpolants(const ParCoordsArrayT& par_coords)
{
  constexpr int nparcoords = ParCoordsArrayT::extent_int(0);
  using res_val_t = std::decay_t<decltype(par_coords(0, 0))>;

  ArrayND<res_val_t[ParCoordsArrayT::extent_int(0)][BasisT::NNODES]> result;
  for (int l = 0; l < nparcoords; ++l) {

    ArrayND<res_val_t[ParCoordsArrayT::extent_int(1)]> point;
    for (int d = 0; d < ParCoordsArrayT::extent_int(1); ++d) {
      point(d) = par_coords(l, d);
    }
    for (int n = 0; n < BasisT::NNODES; ++n) {
      result(l, n) = BasisT::interpolant(n, point);
    }
  }
  return result;
}

template <typename BasisT, typename ParCoordsArrayT>
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr ArrayND<
  typename ParCoordsArrayT::value_type[ParCoordsArrayT::extent_int(0)]
                                      [BasisT::NNODES][BasisT::DIM]>
deriv_coeffs(const ParCoordsArrayT& par_coords)
{
  constexpr int nparcoords = ParCoordsArrayT::extent_int(0);
  using res_val_t = std::decay_t<decltype(par_coords(0, 0))>;

  ArrayND<
    res_val_t[ParCoordsArrayT::extent_int(0)][BasisT::NNODES][BasisT::DIM]>
    result;
  for (int l = 0; l < nparcoords; ++l) {

    ArrayND<res_val_t[ParCoordsArrayT::extent_int(1)]> point;
    for (int d = 0; d < ParCoordsArrayT::extent_int(1); ++d) {
      point(d) = par_coords(l, d);
    }
    for (int n = 0; n < BasisT::NNODES; ++n) {
      for (int d = 0; d < BasisT::DIM; ++d) {
        result(l, n, d) = BasisT::deriv_coeff(n, point, d);
      }
    }
  }
  return result;
}

} // namespace sierra::nalu::utils

#endif