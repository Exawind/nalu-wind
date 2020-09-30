// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUM_DIAGONAL_H
#define MOMENTUM_DIAGONAL_H

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinSysInfo.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {

template <int p>
struct advdiff_diagonal_t
{
  static void invoke(
    double gamma,
    const_elem_offset_view<p> offsets,
    const_scalar_view<p> volumes,
    const_scs_scalar_view<p> adv_metric,
    const_scs_vector_view<p> diff_metric,
    tpetra_view_type owned_yout);
};
} // namespace impl
P_INVOKEABLE(advdiff_diagonal)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
