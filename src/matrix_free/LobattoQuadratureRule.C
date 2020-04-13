// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LobattoQuadratureRule.h"

#include "Kokkos_Array.hpp"
#include "matrix_free/PolynomialOrders.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
constexpr Kokkos::Array<double, 2> GLL<1>::nodes;
constexpr Kokkos::Array<double, 3> GLL<2>::nodes;
constexpr Kokkos::Array<double, 4> GLL<3>::nodes;
constexpr Kokkos::Array<double, 5> GLL<4>::nodes;

double
gauss_lobatto_legendre_abscissae(int p, int n)
{
  switch (p) {
  case inst::P1:
    return GLL<1>::nodes[n];
  case inst::P2:
    return GLL<2>::nodes[n];
  case inst::P3:
    return GLL<3>::nodes[n];
  default:
    return GLL<4>::nodes[n];
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
