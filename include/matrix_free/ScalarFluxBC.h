#ifndef SCALAR_FLUX_BC_H
#define SCALAR_FLUX_BC_H

#include <Tpetra_MultiVector_decl.hpp>

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

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
