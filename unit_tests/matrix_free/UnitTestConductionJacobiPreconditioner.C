// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionJacobiPreconditioner.h"

#include "StkConductionFixture.h"
#include "gtest/gtest.h"

#include "matrix_free/ConductionFields.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/StkToTpetraMap.h"

#include "stk_mesh/base/MetaData.hpp"
#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include <stddef.h>
#include <type_traits>
#include <algorithm>

namespace sierra {
namespace nalu {
namespace matrix_free {

class JacobiFixture : public ConductionFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int nx = 3;
  static constexpr double scale = nx;

  JacobiFixture()
    : ConductionFixture(nx, scale),
      owned_map(make_owned_row_map(mesh, meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh, meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      owned_lhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_rhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_and_shared_lhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      owned_and_shared_rhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh,
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh, meta.universal_part()))
  {
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  const Tpetra::Export<> exporter;
  Tpetra::MultiVector<> owned_lhs;
  Tpetra::MultiVector<> owned_rhs;
  Tpetra::MultiVector<> owned_and_shared_lhs;
  Tpetra::MultiVector<> owned_and_shared_rhs;

  const const_entity_row_view_type elid;
  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;
};

TEST_F(JacobiFixture, jacobi_operator_is_stricly_positive_for_laplacian)
{
  auto fields = gather_required_conduction_fields<order>(meta, conn);
  LinearizedResidualFields<order> coefficient_fields;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  JacobiOperator<order> jac_op(offsets, exporter);
  jac_op.set_coefficients(0.0, coefficient_fields);
  jac_op.compute_diagonal();
  auto& result = jac_op.get_inverse_diagonal();
  auto view_h = result.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t k = 0u; k < result.getLocalLength(); ++k) {
    ASSERT_GT(view_h(k, 0), 1.0e-2);
  }
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
