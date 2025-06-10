// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "StkConductionFixture.h"
#include "matrix_free/SparsifiedEdgeLaplacian.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Teuchos_RCP.hpp"
#include "Tpetra_FECrsGraph.hpp"
#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Assembly_Helpers.hpp"

#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"

#include <math.h>
#include <memory>
#include <vector>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace sparsfied_edge_test {

constexpr int edge_conn[12][2][3] = {
  // bottom face
  {{0, 0, 0}, {0, 0, 1}},
  {{0, 0, 1}, {0, 1, 1}},
  {{0, 1, 1}, {0, 1, 0}},
  {{0, 1, 0}, {0, 0, 0}},

  // top face
  {{1, 0, 0}, {1, 0, 1}},
  {{1, 0, 1}, {1, 1, 1}},
  {{1, 1, 1}, {1, 1, 0}},
  {{1, 1, 0}, {1, 0, 0}},

  // edges from bottom to top
  {{0, 0, 0}, {1, 0, 0}},
  {{0, 0, 1}, {1, 0, 1}},
  {{0, 1, 1}, {1, 1, 1}},
  {{0, 1, 0}, {1, 1, 0}},
};

template <int p>
Teuchos::RCP<Tpetra::FECrsMatrix<>>
create_edge_matrix(
  const StkToTpetraMaps& linsys, const_elem_offset_view<p> offsets)
{
  auto params = Teuchos::rcp(new Teuchos::ParameterList());
  auto owned = Teuchos::rcpFromRef(linsys.owned);
  auto owned_and_shared = Teuchos::rcpFromRef(linsys.owned_and_shared);

  constexpr int star_stencil = 7;
  auto graph = Teuchos::rcp(new Tpetra::FECrsGraph<>(
    owned, owned_and_shared, owned_and_shared, star_stencil, owned_and_shared,
    Teuchos::null, owned, owned, params));
  params->set("Check Col GIDs In At Least One Owned Row", false);

  auto offsets_h = Kokkos::create_mirror_view(offsets);
  Kokkos::deep_copy(offsets_h, offsets);

  using host_policy = Kokkos::MDRangePolicy<
    Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<5>, int>;
  Tpetra::beginAssembly(*graph);
  auto range =
    host_policy({0, 0, 0, 0, 0}, {offsets_h.extent_int(0), p, p, p, 12});
  Kokkos::parallel_for(range, [&](int index, int n, int m, int l, int iedge) {
    for (int nsimd = 0; nsimd < simd_len; ++nsimd) {
      Kokkos::Array<typename Tpetra::Map<>::local_ordinal_type, 2> ids;
      for (int lr = 0; lr < 2; ++lr) {
        const auto sub_n_index = n + edge_conn[iedge][lr][0];
        const auto sub_m_index = m + edge_conn[iedge][lr][1];
        const auto sub_l_index = l + edge_conn[iedge][lr][2];
        ids[lr] =
          offsets_h(index, sub_n_index, sub_m_index, sub_l_index, nsimd);
      }
      graph->insertLocalIndices(ids[0], 2, ids.data());
      graph->insertLocalIndices(ids[1], 2, ids.data());
    }
  });
  Tpetra::endAssembly(*graph);
  return Teuchos::rcp(new Tpetra::FECrsMatrix<>(graph));
}

} // namespace sparsfied_edge_test

class SparsifiedEdgeLaplacianFixture : public ::ConductionFixture
{
protected:
  SparsifiedEdgeLaplacianFixture()
    : ConductionFixture(nx, scale),
      linsys(
        stk::mesh::get_updated_ngp_mesh(bulk),
        meta.universal_part(),
        gid_field_ngp),
      offsets(
        create_offset_map<order>(
          mesh, meta.universal_part(), linsys.stk_lid_to_tpetra_lid)),
      mat(sparsfied_edge_test::create_edge_matrix<order>(linsys, offsets))
  {
    static_assert(nx * nx * nx % simd_len == 0, "");
  }
  StkToTpetraMaps linsys;
  const_elem_offset_view<order> offsets;
  Teuchos::RCP<Tpetra::FECrsMatrix<>> mat;
  static constexpr int nx = 4;
  static constexpr double scale = nx;
};

TEST_F(SparsifiedEdgeLaplacianFixture, laplacian_is_an_l_matrix)
{
  if (bulk.parallel_size() > 1) {
    GTEST_SKIP();
  }

  auto& coords = coordinate_field();
  auto coords_ngp = stk::mesh::get_updated_ngp_field<double>(coords);

  Tpetra::beginAssembly(*mat);
  assemble_sparsified_edge_laplacian(
    order, mesh, meta.universal_part(), coords_ngp,
    NoAuraDeviceMatrix(
      mat->getLocalNumRows(), mat->getLocalMatrixDevice(), {},
      linsys.stk_lid_to_tpetra_lid, linsys.stk_lid_to_tpetra_lid));
  Tpetra::endAssembly(*mat);

  for (unsigned i = 0; i < mat->getLocalNumRows(); ++i) {
    auto local_mat = mat->getLocalMatrixHost();
    auto row = local_mat.row(i);
    for (int j = 0; j < row.length; ++j) {
      if (static_cast<unsigned>(row.colidx(j)) == i) {
        ASSERT_TRUE(row.value(j) > 0);
      } else {
        ASSERT_TRUE(row.value(j) <= 0);
      }
    }
  }
}

TEST_F(SparsifiedEdgeLaplacianFixture, row_and_column_sums_are_zero)
{
  if (bulk.parallel_size() > 1) {
    GTEST_SKIP();
  }

  auto& coords = coordinate_field();
  auto coords_ngp = stk::mesh::get_updated_ngp_field<double>(coords);

  Tpetra::beginAssembly(*mat);
  assemble_sparsified_edge_laplacian(
    order, mesh, meta.universal_part(), coords_ngp,
    NoAuraDeviceMatrix(
      mat->getLocalNumRows(), mat->getLocalMatrixDevice(), {},
      linsys.stk_lid_to_tpetra_lid, linsys.stk_lid_to_tpetra_lid));
  Tpetra::endAssembly(*mat);

  Tpetra::Vector<> ones(Teuchos::rcpFromRef(linsys.owned));
  ones.putScalar(1.);
  Tpetra::Vector<> result(Teuchos::rcpFromRef(linsys.owned));
  result.randomize(); // fuzz just in case
  mat->apply(ones, result, Teuchos::NO_TRANS);
  ASSERT_DOUBLE_EQ(result.norm1(), 0);
  result.randomize();
  mat->apply(ones, result, Teuchos::TRANS);
  ASSERT_DOUBLE_EQ(result.norm1(), 0);
}

TEST_F(SparsifiedEdgeLaplacianFixture, sample_for_positive_definiteness)
{
  if (bulk.parallel_size() > 1) {
    return;
  }

  auto& coords = coordinate_field();
  auto coords_ngp = stk::mesh::get_updated_ngp_field<double>(coords);

  Tpetra::beginAssembly(*mat);
  assemble_sparsified_edge_laplacian(
    order, mesh, meta.universal_part(), coords_ngp,
    NoAuraDeviceMatrix(
      mat->getLocalNumRows(), mat->getLocalMatrixDevice(), {},
      linsys.stk_lid_to_tpetra_lid, linsys.stk_lid_to_tpetra_lid));
  Tpetra::endAssembly(*mat);

  Tpetra::Vector<> ones(Teuchos::rcpFromRef(linsys.owned));
  ones.randomize();
  Tpetra::Vector<> result(Teuchos::rcpFromRef(linsys.owned));
  for (int n = 0; n < 20; ++n) {
    result.randomize(); // fuzz just in case
    mat->apply(ones, result);
    ASSERT_GT(ones.dot(result), 0);
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
