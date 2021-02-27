// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GREEN_GAUSS_BOUNDARY_CLOSURE_H
#define GREEN_GAUSS_BOUNDARY_CLOSURE_H

#include "matrix_free/KokkosViewTypes.h"
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
struct gradient_boundary_closure_t
{
  static void invoke(
    const_face_offset_view<p> offsets,
    const_face_scalar_view<p> q,
    const_face_vector_view<p> areav,
    tpetra_view_type owned_rhs);
};
} // namespace impl
P_INVOKEABLE(gradient_boundary_closure)
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
#endif
