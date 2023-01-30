// Copy1 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain 1s in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/SparsifiedEdgeLaplacian.h"

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

namespace {

template <typename LHSArray>
KOKKOS_FUNCTION void
symmetric_scatter(LHSArray& lhs, int iedge, const ftype& dfdq)
{
  lhs(iedge, 0, 0) = dfdq;
  lhs(iedge, 0, 1) = -dfdq;

  lhs(iedge, 1, 0) = -dfdq;
  lhs(iedge, 1, 1) = dfdq;
}

template <int p, typename BoxArrayType, typename LHSArray>
KOKKOS_FUNCTION void
sparsified_laplacian_edge_lhs(const BoxArrayType& box, LHSArray& lhs)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  constexpr double dv[2] = {-0.5, +0.5};
  constexpr int first_order = 1;

  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      const auto metric = geom::laplacian_metric<first_order, XH>(box, 0, k, j);
      const auto dfdq =
        -0.5 * (metric[0] + dv[j] * metric[1] + dv[k] * metric[2]);
      symmetric_scatter(lhs, 2 * k + j, dfdq);
    }
  }

  for (int k = 0; k < 2; ++k) {
    for (int i = 0; i < 2; ++i) {
      const auto metric = geom::laplacian_metric<first_order, YH>(box, 0, k, i);
      const auto dfdq =
        -0.5 * (metric[0] + dv[i] * metric[1] + dv[k] * metric[2]);
      symmetric_scatter(lhs, 4 + 2 * k + i, dfdq);
    }
  }

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const auto metric = geom::laplacian_metric<first_order, ZH>(box, 0, j, i);
      const auto dfdq =
        -0.5 * (metric[0] + dv[i] * metric[1] + dv[j] * metric[2]);
      symmetric_scatter(lhs, 8 + 2 * j + i, dfdq);
    }
  }
}

template <typename RowViewType>
KOKKOS_FUNCTION void
sum_into_row_edge(
  int left, int right, double lhs_l, double lhs_r, RowViewType row)
{
  int offset = 0;
  while (row.colidx(offset) != left && offset < row.length) {
    ++offset;
  }
  Kokkos::atomic_add(&row.value(offset++), lhs_l);

  while (row.colidx(offset) != right && offset < row.length) {
    ++offset;
  }
  Kokkos::atomic_add(&row.value(offset), lhs_r);
}

KOKKOS_FUNCTION void
sum_edge_contribution_into_matrix(
  int left,
  int right,
  const Kokkos::Array<Kokkos::Array<double, 2>, 2>& lhs,
  NoAuraDeviceMatrix mat)
{
  Kokkos::Array<int, 2> collids = {
    {mat.col_lid_map_[left], mat.col_lid_map_[right]}};

  Kokkos::Array<int, 2> rowlids = {
    {mat.row_lid_map_[left], mat.row_lid_map_[right]}};

  const bool is_reversed = collids[0] > collids[1];
  const Kokkos::Array<int, 2> perm = {{int(is_reversed), 1 - int(is_reversed)}};

  for (int n = 0; n < 2; ++n) {
    const int perm_index = perm[n];
    const auto rowlid = rowlids[perm_index];
    if (rowlid < mat.max_owned_row_) {
      sum_into_row_edge(
        collids[perm[0]], collids[perm[1]], lhs[perm[n]][perm[0]],
        lhs[perm[n]][perm[1]], mat.owned_mat_.row(rowlid));
    } else {
      sum_into_row_edge(
        collids[perm[0]], collids[perm[1]], lhs[perm[n]][perm[0]],
        lhs[perm[n]][perm[1]],
        mat.shared_mat_.row(rowlid - mat.max_owned_row_));
    }
  }
}

} // namespace

template <int p>
void
assemble_sparsified_edge_laplacian_t<p>::invoke(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  const stk::mesh::NgpField<double>& coords,
  NoAuraDeviceMatrix mat)
{
  // map from the edge ordinals to an order
  // convenient for computing the edge laplacian
  constexpr LocalArray<int[12][2][3]> edges = {{
    {{0, 0, 0}, {0, 0, 1}}, // {0,1} .
    {{0, 1, 0}, {0, 1, 1}}, // {3,2} .
    {{1, 0, 0}, {1, 0, 1}}, // {4,5} .
    {{1, 1, 0}, {1, 1, 1}}, // {7,6} .
    {{0, 0, 0}, {0, 1, 0}}, // {0,3}
    {{0, 0, 1}, {0, 1, 1}}, // {1,2}
    {{1, 0, 0}, {1, 1, 0}}, // {4,7}
    {{1, 0, 1}, {1, 1, 1}}, // {5,6}
    {{0, 0, 0}, {1, 0, 0}}, // {0,4}
    {{0, 0, 1}, {1, 0, 1}}, // {1,5}
    {{0, 1, 0}, {1, 1, 0}}, // {3,7}
    {{0, 1, 1}, {1, 1, 1}}, // {2,6}
  }};

  const auto conn = stk_connectivity_map<p>(mesh, active);
  vector_view<p> xc{"coords", conn.extent(0)};
  field_gather<p>(conn, coords, xc);

  Kokkos::parallel_for(
    DeviceRangePolicy(0, conn.extent_int(0)), KOKKOS_LAMBDA(int index) {
      auto elem_coords = Kokkos::subview(
        xc, index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
      const auto length = valid_offset<p>(index, conn);
      for (int n = 0; n < p; ++n) {
        for (int m = 0; m < p; ++m) {
          for (int l = 0; l < p; ++l) {
            const auto box = hex_vertex_coordinates(n, m, l, elem_coords);
            LocalArray<ftype[12][2][2]> edge_lhs;
            sparsified_laplacian_edge_lhs<p>(box, edge_lhs);
            for (int e = 0; e < 12; ++e) {
              const int ln = n + edges(e, 0, 0);
              const int rn = n + edges(e, 1, 0);
              const int lm = m + edges(e, 0, 1);
              const int rm = m + edges(e, 1, 1);
              const int ll = l + edges(e, 0, 2);
              const int rl = l + edges(e, 1, 2);
              for (int nsimd = 0; nsimd < length; ++nsimd) {
                Kokkos::Array<Kokkos::Array<double, 2>, 2> lhs;
                lhs[0][0] = stk::simd::get_data(edge_lhs(e, 0, 0), nsimd);
                lhs[0][1] = stk::simd::get_data(edge_lhs(e, 0, 1), nsimd);
                lhs[1][0] = stk::simd::get_data(edge_lhs(e, 1, 0), nsimd);
                lhs[1][1] = stk::simd::get_data(edge_lhs(e, 1, 1), nsimd);

                const auto left = mesh.get_entity(
                  stk::topology::NODE_RANK, conn(index, ln, lm, ll, nsimd));
                const auto right = mesh.get_entity(
                  stk::topology::NODE_RANK, conn(index, rn, rm, rl, nsimd));
                sum_edge_contribution_into_matrix(
                  left.local_offset(), right.local_offset(), lhs, mat);
              }
            }
          }
        }
      }
    });
}
INSTANTIATE_POLYSTRUCT(assemble_sparsified_edge_laplacian_t);
} // namespace impl
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
