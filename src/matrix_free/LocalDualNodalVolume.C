// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/LocalDualNodalVolume.h"

#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/ValidSimdLength.h"

#include "matrix_free/TensorOperations.h"
#include "matrix_free/GeometricFunctions.h"

#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include <KokkosInterface.h>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace impl {

template <int p, typename BoxArray, typename OutArray>
KOKKOS_FORCEINLINE_FUNCTION void
compute_volumes(const BoxArray& box, OutArray& out)
{
  static constexpr auto vandermonde = Coeffs<p>::W;

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      LocalArray<ftype[p + 1]> scratch;
      for (int i = 0; i < p + 1; ++i) {
        scratch(i) =
          determinant<ftype>(geom::linear_hex_jacobian<p>(box, k, j, i));
      }
      for (int i = 0; i < p + 1; ++i) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(i, q) * scratch(q);
        }
        out(k, j, i) = acc;
      }
    }
  }

  for (int i = 0; i < p + 1; ++i) {
    LocalArray<ftype[p + 1][p + 1]> scratch;
    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(j, q) * out(k, q, i);
        }
        scratch(k, j) = acc;
      }
    }

    for (int k = 0; k < p + 1; ++k) {
      for (int j = 0; j < p + 1; ++j) {
        ftype acc(0);
        for (int q = 0; q < p + 1; ++q) {
          acc += vandermonde(k, q) * scratch(q, j);
        }
        out(k, j, i) = acc;
      }
    }
  }
}

template <int p>
void
local_dual_nodal_volume_t<p>::invoke(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  const stk::mesh::NgpField<double>& coords,
  stk::mesh::NgpField<double>& dnv)
{
  const auto conn = stk_connectivity_map<p>(mesh, sel);
  vector_view<p> xc{"coords", conn.extent(0)};
  field_gather<p>(conn, coords, xc);

  dnv.set_all(mesh, 0.);
  Kokkos::parallel_for(
    DeviceRangePolicy(0, xc.extent_int(0)), KOKKOS_LAMBDA(int index) {
      const auto box = hex_vertex_coordinates<p>(index, xc);
      const auto valid_length = valid_offset<p>(index, conn);

      LocalArray<ftype[p + 1][p + 1][p + 1]> vols;
      compute_volumes<p>(box, vols);

      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < valid_length; ++n) {
              const auto mi = conn(index, k, j, i, n);
              Kokkos::atomic_add(
                &dnv.get(mi, 0), stk::simd::get_data(vols(k, j, i), n));
            }
          }
        }
      }
    });
  dnv.modify_on_device();
}

INSTANTIATE_POLYSTRUCT(local_dual_nodal_volume_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
