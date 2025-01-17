// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONTINUITY_INTERIOR_H
#define CONTINUITY_INTERIOR_H

#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/PolynomialOrders.h"

#include "Teuchos_RCP.hpp"
#include "Tpetra_MultiVector.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using ra_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread;

namespace impl {
template <int p>
struct continuity_residual_t
{
  using narray = ArrayND<ftype[p + 1][p + 1][p + 1]>;

  static void invoke(
    double scaling,
    const_elem_offset_view<p> offsets,
    const_scs_scalar_view<p> mdot,
    tpetra_view_type owned_rhs);
};
} // namespace impl
P_INVOKEABLE(continuity_residual)

namespace impl {
template <int p>
struct continuity_linearized_residual_t
{
  using narray = ArrayND<ftype[p + 1][p + 1][p + 1]>;

  static void invoke(
    const_elem_offset_view<p> offsets,
    const_scs_vector_view<p> metric,
    ra_tpetra_view_type xin,
    tpetra_view_type yout);
};
} // namespace impl
P_INVOKEABLE(continuity_linearized_residual)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
