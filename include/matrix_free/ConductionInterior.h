// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_INTERIOR_H
#define CONDUCTION_INTERIOR_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

#include "Kokkos_Array.hpp"
#include "Tpetra_MultiVector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

namespace impl {
template <int p>
struct conduction_residual_t
{
  using narray = ArrayND<ftype[p + 1][p + 1][p + 1]>;

  static void invoke(
    Kokkos::Array<double, 3> gammas,
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> qm1,
    const_scalar_view<p> qp0,
    const_scalar_view<p> qp1,
    const_scalar_view<p> volume_metric,
    const_scs_vector_view<p> diffusion_metric,
    tpetra_view_type owned_rhs);
};
} // namespace impl
P_INVOKEABLE(conduction_residual)
namespace impl {
template <int p>
struct conduction_linearized_residual_t
{
  using narray = ArrayND<ftype[p + 1][p + 1][p + 1]>;

  static void invoke(
    double gamma,
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> volume_metric,
    const_scs_vector_view<p> diffusion_metric,
    ra_tpetra_view_type delta_owned,
    tpetra_view_type rhs);
};
} // namespace impl
P_INVOKEABLE(conduction_linearized_residual)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
