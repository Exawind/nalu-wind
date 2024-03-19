// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MomentumJacobi.h"

#include "StkLowMachFixture.h"
#include "gtest/gtest.h"

#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinearAdvectionMetric.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearDiffusionMetric.h"
#include "matrix_free/LowMachFields.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/StkToTpetraMap.h"

#include "Kokkos_Core.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"

#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_topology/topology.hpp"
#include "stk_math/StkMath.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"

#include <iostream>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

class MomentumJacobiOperatorFixture : public LowMachFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);

  MomentumJacobiOperatorFixture()
    : LowMachFixture(nx, scale),
      owned_map(make_owned_row_map(mesh(), active())),
      owned_and_shared_map(
        make_owned_and_shared_row_map(mesh(), active(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      lhs(Teuchos::rcpFromRef(owned_map), 3),
      rhs(Teuchos::rcpFromRef(owned_map), 3),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh(), active(), gid_field_ngp, owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh(), active())),
      offsets(create_offset_map<order>(mesh(), active(), elid))
  {
    lhs.putScalar(0.);
    rhs.putScalar(0.);
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  Tpetra::Export<> exporter;
  Tpetra::MultiVector<> lhs;
  Tpetra::MultiVector<> rhs;
  const const_entity_row_view_type elid;

  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;

  static constexpr int nx = 4;
  static constexpr double scale = 1;
};

TEST_F(MomentumJacobiOperatorFixture, diagonal_positive)
{
  MomentumJacobiOperator<order> prec_op(offsets, exporter);

  auto fields = gather_required_lowmach_fields<order>(meta, conn);
  prec_op.compute_diagonal(
    1., fields.volume_metric, fields.advection_metric, fields.diffusion_metric);

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
