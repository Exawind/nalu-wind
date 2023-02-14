// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/FilterJacobi.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/StkGradientFixture.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/StkToTpetraMap.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "gtest/gtest.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_simd/Simd.hpp"

#include <stddef.h>
#include <cmath>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

class FilterJacobiFixture : public GradientFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);
  static constexpr int nx = 3;
  static constexpr double scale = nx;

  FilterJacobiFixture()
    : GradientFixture(nx, scale),
      owned_map(make_owned_row_map(mesh(), meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh(), meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      owned_lhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_rhs(Teuchos::rcpFromRef(owned_map), 1),
      owned_and_shared_lhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      owned_and_shared_rhs(Teuchos::rcpFromRef(owned_and_shared_map), 1),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh(),
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh(), meta.universal_part())),
      offsets(create_offset_map<order>(mesh(), active(), elid))
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

TEST_F(FilterJacobiFixture, jacobi_operator_is_stricly_positive_for_mass)
{
  vector_view<order> elem_coords("elem_coords", conn.extent(0));
  auto coord_ngp = stk::mesh::get_updated_ngp_field<double>(coordinate_field());
  field_gather<order>(conn, coord_ngp, elem_coords);
  const auto vols = geom::volume_metric<order>(elem_coords);

  FilterJacobiOperator<order> prec_op(offsets, exporter);
  prec_op.compute_diagonal(vols);

  auto& result = prec_op.get_inverse_diagonal();
  auto view_h = result.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t k = 0u; k < result.getLocalLength(); ++k) {
    ASSERT_TRUE(std::isfinite(stk::simd::get_data(view_h(k, 0), 0)));
    ASSERT_GT(view_h(k, 0), 1.0e-2);
  }
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
