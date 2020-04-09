#ifndef CONDUCTION_INTERIOR_H
#define CONDUCTION_INTERIOR_H

#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>
#include <Kokkos_Array.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_MultiVector_decl.hpp>

#include "Tpetra_MultiVector.hpp"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/LocalArray.h"

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
  using narray = LocalArray<ftype[p + 1][p + 1][p + 1]>;

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
  using narray = LocalArray<ftype[p + 1][p + 1][p + 1]>;

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
