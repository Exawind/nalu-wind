// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "StkConductionFixture.h"
#include "gtest/gtest.h"

#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionInfo.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/StrongDirichletBC.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/StkToTpetraMap.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_CombineMode.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_topology/topology.hpp"

#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

class DirichletFixture : public ConductionFixture
{
protected:
  DirichletFixture()
    : ConductionFixture(nx, scale),
      owned_map(make_owned_row_map(mesh, meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh, meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      owned_lhs(Teuchos::rcpFromRef(owned_map), 3),
      owned_rhs(Teuchos::rcpFromRef(owned_map), 3),
      owned_and_shared_lhs(Teuchos::rcpFromRef(owned_and_shared_map), 3),
      owned_and_shared_rhs(Teuchos::rcpFromRef(owned_and_shared_map), 3),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh,
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      dirichlet_nodes(simd_node_map(
        mesh, meta.get_topology_root_part(stk::topology::QUAD_4))),
      dirichlet_offsets(simd_node_offsets(
        mesh, meta.get_topology_root_part(stk::topology::QUAD_4), elid))
  {
    owned_lhs.putScalar(0.);
    owned_rhs.putScalar(0.);

    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) = 5.0;
        *stk::mesh::field_data(qbc_field, node) = -2.3;
      }
    }
  }

  static constexpr int nx = 4;
  static constexpr double scale = M_PI;

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  const Tpetra::Export<> exporter;
  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;
  Tpetra::MultiVector<> owned_and_shared_lhs;
  Tpetra::MultiVector<> owned_and_shared_rhs;

  const const_entity_row_view_type elid;
  const const_node_mesh_index_view dirichlet_nodes;
  const const_node_offset_view dirichlet_offsets;
};

TEST_F(DirichletFixture, bc_residual_scalar)
{
  auto qp1 = node_scalar_view("qp1_at_bc", dirichlet_nodes.extent_int(0));
  Kokkos::deep_copy(qp1, 5);

  auto qbc =
    node_scalar_view("qspecified_at_bc", dirichlet_nodes.extent_int(0));
  Kokkos::deep_copy(qbc, -2.3);

  owned_and_shared_rhs.putScalar(0.);
  dirichlet_residual(
    dirichlet_offsets, qp1, qbc, owned_rhs.getLocalLength(),
    owned_and_shared_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(owned_and_shared_rhs, exporter, Tpetra::ADD);

  auto view_h = owned_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    maxval = std::max(maxval, std::abs(view_h(k, 0)));
  }
  ASSERT_DOUBLE_EQ(maxval, 7.3);
}

TEST_F(DirichletFixture, bc_residual_vector)
{
  auto qp1 = node_vector_view("qp1_at_bc", dirichlet_nodes.extent_int(0));
  Kokkos::deep_copy(qp1, 5.0);

  auto qbc = node_vector_view("qbc", dirichlet_nodes.extent_int(0));
  Kokkos::deep_copy(qbc, -2.3);

  owned_and_shared_rhs.putScalar(0.);
  dirichlet_residual(
    dirichlet_offsets, qp1, qbc, owned_rhs.getLocalLength(),
    owned_and_shared_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(owned_and_shared_rhs, exporter, Tpetra::ADD);

  auto view_h = owned_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    for (int d = 0; d < 3; ++d) {
      maxval = std::max(maxval, std::abs(view_h(k, d)));
    }
  }
  ASSERT_DOUBLE_EQ(maxval, 7.3);
}

TEST_F(DirichletFixture, linearized_bc_residual)
{
  constexpr double some_val = 85432.2;
  owned_lhs.putScalar(some_val);

  owned_and_shared_lhs.doImport(owned_lhs, exporter, Tpetra::INSERT);

  dirichlet_linearized(
    dirichlet_offsets, owned_lhs.getLocalLength(),
    owned_and_shared_lhs.getLocalViewDevice(Tpetra::Access::ReadWrite),
    owned_and_shared_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite));
  owned_rhs.putScalar(0.);
  owned_rhs.doExport(owned_and_shared_rhs, exporter, Tpetra::ADD);

  auto view_h = owned_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  constexpr double tol = 1.0e-14;
  double maxval = -1;
  for (size_t k = 0u; k < owned_rhs.getLocalLength(); ++k) {
    const bool zero_or_val = std::abs(view_h(k, 0) - 0) < tol ||
                             std::abs(view_h(k, 0) - some_val) < tol;
    ASSERT_TRUE(zero_or_val);
    maxval = std::max(maxval, view_h(k, 0));
  }
  ASSERT_DOUBLE_EQ(maxval, some_val);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
