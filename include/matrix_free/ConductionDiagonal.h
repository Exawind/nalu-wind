// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONDUCTION_DIAGONAL_H
#define CONDUCTION_DIAGONAL_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"

#include <Tpetra_MultiVector.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {
using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;
using const_tpetra_view_type =
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const;
namespace impl {

template <int p>
struct conduction_diagonal_t
{
  static void invoke(
    double gamma,
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> volumes,
    const_scs_vector_view<p> metric,
    tpetra_view_type owned_yout);
};
} // namespace impl
P_INVOKEABLE(conduction_diagonal)

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
