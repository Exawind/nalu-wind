// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GreenGaussGradientOperator.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraLocalIndices.h"
#include "matrix_free/StkToTpetraMap.h"

#include "StkGradientFixture.h"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_RCP.hpp"

#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldTraits.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_topology/topology.hpp"

#include <math.h>
#include <algorithm>
#include <random>
#include <stddef.h>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

class GradientOperatorFixture : public GradientFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);

  GradientOperatorFixture()
    : GradientFixture(nx, scale),
      owned_map(make_owned_row_map(mesh(), meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh(), meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      lhs(Teuchos::rcpFromRef(owned_map), 3),
      rhs(Teuchos::rcpFromRef(owned_map), 3),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh(),
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh(), meta.universal_part())),
      offsets(create_offset_map<order>(mesh(), meta.universal_part(), elid))
  {
    lhs.putScalar(0.);
    rhs.putScalar(0.);
  }

  GradientResidualFields<order> gather_required_fields()
  {
    GradientResidualFields<order> fields;
    fields.q = scalar_view<order>("q", offsets.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(q_field), fields.q);

    fields.dqdx = vector_view<order>("dqdx", offsets.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(dqdx_field), fields.dqdx);

    const auto coords = vector_view<order>("coords", offsets.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(coordinate_field()),
      coords);
    fields.vols = geom::volume_metric<order>(coords);
    fields.areas = geom::linear_areas<order>(coords);
    return fields;
  }

  void randomize_fields()
  {
    std::mt19937 rng;
    rng.seed(0); // fixed seed
    std::uniform_real_distribution<double> coeff(1.0, 2.0);

    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(q_field, node) = coeff(rng);
        stk::mesh::field_data(dqdx_field, node)[0] = coeff(rng);
        stk::mesh::field_data(dqdx_field, node)[1] = coeff(rng);
        stk::mesh::field_data(dqdx_field, node)[2] = coeff(rng);
      }
    }
    stk::mesh::copy_owned_to_shared(bulk, {&q_field, &dqdx_field});
  }

  void zero_fields()
  {
    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        *stk::mesh::field_data(q_field, node) = 0.0;
        stk::mesh::field_data(dqdx_field, node)[0] = 0.0;
        stk::mesh::field_data(dqdx_field, node)[1] = 0.0;
        stk::mesh::field_data(dqdx_field, node)[2] = 0.0;
      }
    }
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  Tpetra::Export<> exporter;
  Tpetra::MultiVector<> lhs;
  Tpetra::MultiVector<> rhs;
  const const_entity_row_view_type elid;

  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;
  static constexpr int nx = 3;
  static constexpr double scale = M_PI;
};

TEST_F(GradientOperatorFixture, residual_operator_zero_for_zero_data)
{
  zero_fields();
  GradientResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields(gather_required_fields());
  resid_op.compute(rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    ASSERT_NEAR(view_h(k, 0), 0, 1.0e-14);
  }
}

TEST_F(GradientOperatorFixture, residual_operator_not_zero_for_nonconstant_data)
{
  randomize_fields();
  GradientResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields(gather_required_fields());
  resid_op.compute(rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
  double max_error = -1;
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    max_error = std::max(std::abs(view_h(k, 0)), max_error);
  }
  ASSERT_TRUE(max_error > 1.0e-8);
}

TEST_F(
  GradientOperatorFixture, linearized_residual_operator_zero_for_constant_data)
{
  zero_fields();
  auto fields = gather_required_fields();
  GradientLinearizedResidualOperator<order> lin_op(offsets, exporter);
  lin_op.set_volumes(fields.vols);

  lhs.putScalar(0.);
  lin_op.apply(lhs, rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    ASSERT_NEAR(view_h(k, 0), 0, 1.e-14);
  }
}

TEST_F(
  GradientOperatorFixture,
  linearized_residual_operator_not_zero_for_nonconstant_data)
{
  randomize_fields();
  auto fields = gather_required_fields();
  GradientLinearizedResidualOperator<order> lin_op(offsets, exporter);
  lin_op.set_volumes(fields.vols);

  lhs.randomize(-1, +1);

  lin_op.apply(lhs, rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  double max_error = -1;
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    max_error = std::max(std::abs(view_h(k, 0)), max_error);
  }
  ASSERT_TRUE(max_error > 1.0e-8);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
