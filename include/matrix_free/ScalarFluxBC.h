// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SCALAR_FLUX_BC_H
#define SCALAR_FLUX_BC_H

#include <Tpetra_MultiVector.hpp>

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

using tpetra_view_type = typename Tpetra::MultiVector<>::dual_view_type::t_dev;

namespace impl {
template <int p>
struct scalar_neumann_residual_t
{
  static void invoke(
    const_face_offset_view<p> offsets,
    const_face_scalar_view<p> dqdn,
    const_face_vector_view<p> areav,
    tpetra_view_type owned_rhs);
};
} // namespace impl
P_INVOKEABLE(scalar_neumann_residual)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
