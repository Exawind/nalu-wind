// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GAUSS_LEGENDRE_QUADRATURE_RULE
#define GAUSS_LEGENDRE_QUADRATURE_RULE

#include "Kokkos_Array.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
template <int n>
struct LGL
{
};

template <>
struct LGL<1>
{
  static constexpr Kokkos::Array<double, 1> nodes = {{0}};
};

template <>
struct LGL<2>
{
  static constexpr Kokkos::Array<double, 2> nodes = {
    {-0.5773502691896257, +0.5773502691896257}};
};

template <>
struct LGL<3>
{
  static constexpr Kokkos::Array<double, 3> nodes = {
    {-0.7745966692414834, 0, +0.7745966692414834}};
};

template <>
struct LGL<5>
{
  static constexpr Kokkos::Array<double, 5> nodes = {
    {-0.8611363115940526, -0.3399810435848563, 0, +0.3399810435848563,
     +0.8611363115940526}};
};
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
