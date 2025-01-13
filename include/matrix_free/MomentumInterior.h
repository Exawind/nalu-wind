// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUM_INTERIOR_H
#define MOMENTUM_INTERIOR_H

#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/PolynomialOrders.h"

#include "Kokkos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_MultiVector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

namespace impl {

// maybe load visc
template <int p>
struct momentum_residual_t
{
  using narray = ArrayND<ftype[p + 1][p + 1][p + 1]>;

  static void invoke(
    Kokkos::Array<double, 3> gammas,
    const_elem_offset_view<p> offsets,
    const_vector_view<p> xc,
    const_scalar_view<p> rho,
    const_scalar_view<p> visc,
    const_scalar_view<p> vm1,
    const_scalar_view<p> vp0,
    const_scalar_view<p> vp1,
    const_vector_view<p> um1,
    const_vector_view<p> up0,
    const_vector_view<p> up1,
    const_vector_view<p> gp,
    const_vector_view<p> force,
    const_scs_scalar_view<p> mdot,
    tpetra_view_type yout);
};
} // namespace impl
P_INVOKEABLE(momentum_residual)
namespace impl {
template <int p>
struct momentum_linearized_residual_t
{
  using narray = ArrayND<ftype[p + 1][p + 1][p + 1]>;

  static void invoke(
    double proj_time_scale,
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> vp1,
    const_scs_scalar_view<p> mdot,
    const_scs_vector_view<p> diff,
    ra_tpetra_view_type xin,
    tpetra_view_type yout);
};
} // namespace impl
P_INVOKEABLE(momentum_linearized_residual)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
