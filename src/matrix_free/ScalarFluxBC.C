// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ScalarFluxBC.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"
#include "matrix_free/KokkosViewTypes.h"

#include <KokkosInterface.h>
#include "Kokkos_ScatterView.hpp"
#include "Kokkos_Macros.hpp"
#include "Teuchos_RCP.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_simd/Simd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace {
template <
  int p,
  typename FaceRankInput,
  typename AreaArray,
  typename ScratchArray,
  typename FaceRankOutput>
KOKKOS_FUNCTION void
scalar_flux(
  int index,
  const FaceRankInput& in,
  const AreaArray& areav,
  ScratchArray& scratch,
  FaceRankOutput& out)
{
  for (int j = 0; j < p + 1; ++j) {
    for (int i = 0; i < p + 1; ++i) {
      const ftype ax = areav(index, j, i, 0);
      const ftype ay = areav(index, j, i, 1);
      const ftype az = areav(index, j, i, 2);
      out(j, i) =
        in(index, j, i) * stk::math::sqrt(ax * ax + ay * ay + az * az);
    }
  }

  static constexpr auto vandermonde = Coeffs<p>::W;
  for (int j = 0; j < p + 1; ++j) {
    for (int i = 0; i < p + 1; ++i) {
      ftype acc(0);
      for (int q = 0; q < p + 1; ++q) {
        acc += vandermonde(i, q) * out(j, q);
      }
      scratch(j, i) = acc;
    }
  }

  for (int j = 0; j < p + 1; ++j) {
    for (int i = 0; i < p + 1; ++i) {
      out(j, i) = 0;
    }

    for (int q = 0; q < p + 1; ++q) {
      const auto temp = vandermonde(j, q);
      for (int i = 0; i < p + 1; ++i) {
        out(j, i) += temp * scratch(q, i);
      }
    }
  }
}

} // namespace

namespace impl {
template <int p>
void
scalar_neumann_residual_t<p>::invoke(
  const_face_offset_view<p> offsets,
  const_face_scalar_view<p> dqdn,
  const_face_vector_view<p> areav,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("scalar_neumann_residual");
  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    "flux_residual", DeviceRangePolicy(0, offsets.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      LocalArray<ftype[p + 1][p + 1]> element_rhs;
      {
        LocalArray<ftype[p + 1][p + 1]> scratch;
        scalar_flux<p>(index, dqdn, areav, scratch, element_rhs);
      }

      auto accessor = yout_scatter.access();
      const int valid_length = valid_offset<p>(index, offsets);
      for (int j = 0; j < p + 1; ++j) {
        for (int i = 0; i < p + 1; ++i) {
          for (int n = 0; n < valid_length; ++n) {
            accessor(offsets(index, j, i, n), 0) +=
              stk::simd::get_data(element_rhs(j, i), n);
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(scalar_neumann_residual_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
