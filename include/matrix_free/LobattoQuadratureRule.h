// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LOBATTO_QUADRATURE_RULE_H
#define LOBATTO_QUADRATURE_RULE_H

#include "Kokkos_Array.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
template <int n>
struct GLL
{
};

template <>
struct GLL<1>
{
  static constexpr Kokkos::Array<double, 2> nodes = {{-1, +1}};
};

template <>
struct GLL<2>
{
  static constexpr Kokkos::Array<double, 3> nodes = {{-1, 0, +1}};
};

template <>
struct GLL<3>
{
  static constexpr Kokkos::Array<double, 4> nodes = {
    {-1, -0.447213595499957939281834733746, +0.447213595499957939281834733746,
     +1}};
};

template <>
struct GLL<4>
{
  static constexpr Kokkos::Array<double, 5> nodes = {
    {-1, -0.654653670707977143798292456247, 0,
     +0.654653670707977143798292456247, +1}};
};

double gauss_lobatto_legendre_abscissae(int p, int n);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
