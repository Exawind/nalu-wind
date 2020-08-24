// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANT_REYNOLDS_H
#define COURANT_REYNOLDS_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LocalArray.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {
template <int p>
struct max_local_courant_t
{
  static double
  invoke(double dt, const_vector_view<p> xc, const_vector_view<p> vel);
};

template <int p>
struct max_local_reynolds_t
{
  static double invoke(
    const_vector_view<p> xc,
    const_scalar_view<p> rho,
    const_scalar_view<p> visc,
    const_vector_view<p> vel);
};

template <int p>
struct max_local_courant_reynolds_t
{
  static Kokkos::Array<double, 2> invoke(
    double dt,
    const_vector_view<p> xc,
    const_scalar_view<p> rho,
    const_scalar_view<p> visc,
    const_vector_view<p> vel);
};

} // namespace impl
P_INVOKEABLE(max_local_courant)
P_INVOKEABLE(max_local_reynolds)
P_INVOKEABLE(max_local_courant_reynolds)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
