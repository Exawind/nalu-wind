// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FILTER_DIAGONAL_H
#define FILTER_DIAGONAL_H

#include "Tpetra_MultiVector.hpp"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {
template <int p>
struct filter_diagonal_t
{
  static void invoke(
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> vols,
    typename Tpetra::MultiVector<>::dual_view_type::t_dev yout,
    bool lumped = (p == 1));
};
} // namespace impl
P_INVOKEABLE(filter_diagonal)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
