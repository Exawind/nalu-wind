// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussBoundaryClosure.h"

#include "matrix_free/Coefficients.h"
#include "matrix_free/KokkosFramework.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"
#include "Kokkos_ScatterView.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"
#include "stk_simd/Simd.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace impl {

namespace {
template <
  int p,
  typename FaceRankInput,
  typename AreaArray,
  typename FaceRankOutput>
KOKKOS_FUNCTION void
vector_component_flux(
  int index,
  const FaceRankInput& in,
  const AreaArray& areav,
  FaceRankOutput& out,
  int component)
{
  ArrayND<ftype[p + 1][p + 1]> scratch;

  for (int j = 0; j < p + 1; ++j) {
    for (int i = 0; i < p + 1; ++i) {
      out(j, i) = in(index, j, i) * areav(index, j, i, component);
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

template <int p>
void
gradient_boundary_closure_t<p>::invoke(
  const_face_offset_view<p> offsets,
  const_face_scalar_view<p> q,
  const_face_vector_view<p> areav,
  tpetra_view_type yout)
{
  stk::mesh::ProfilingBlock pf("gradient_boundary_closure");
  auto yout_scatter = Kokkos::Experimental::create_scatter_view(yout);
  Kokkos::parallel_for(
    DeviceRangePolicy(0, offsets.extent_int(0)), KOKKOS_LAMBDA(int index) {
      for (int d = 0; d < 3; ++d) {
        ArrayND<ftype[p + 1][p + 1]> rhs;
        vector_component_flux<p>(index, q, areav, rhs, d);

        auto accessor = yout_scatter.access();
        const int valid_length = valid_offset<p>(index, offsets);
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_length; ++n) {
              const auto id = offsets(index, j, i, n);
              accessor(id, d) += stk::simd::get_data(rhs(j, i), n);
            }
          }
        }
      }
    });
  Kokkos::Experimental::contribute(yout, yout_scatter);
}
INSTANTIATE_POLYSTRUCT(gradient_boundary_closure_t);

} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
