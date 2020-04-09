#ifndef CONDUCTION_DIAGONAL_H
#define CONDUCTION_DIAGONAL_H

#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_Operator.hpp>

#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"

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

void dirichlet_diagonal(
  const_node_offset_view offsets,
  int max_owned_lid,
  tpetra_view_type owned_yout);
} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
