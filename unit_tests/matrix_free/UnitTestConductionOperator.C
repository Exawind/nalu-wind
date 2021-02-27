// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionOperator.h"

#include "StkConductionFixture.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/KokkosViewTypes.h"
#include "matrix_free/MakeRCP.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/StkToTpetraLocalIndices.h"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include <Teuchos_RCP.hpp>
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_Map.hpp>

#include "gtest/gtest.h"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_topology/topology.hpp"

#include <algorithm>
#include <random>

namespace sierra {
namespace nalu {
namespace matrix_free {

class ConductionOperatorFixture : public ConductionFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);

  ConductionOperatorFixture()
    : ConductionFixture(nx, scale),
      owned_map(make_owned_row_map(mesh, meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh, meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      lhs(Teuchos::rcpFromRef(owned_map), 1),
      rhs(Teuchos::rcpFromRef(owned_map), 1),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh,
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh, meta.universal_part())),
      offsets(create_offset_map<order>(mesh, meta.universal_part(), elid))
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
  node_offset_view dirichlet_bc_nodes{"dirichlet_bc_nodes_empty", 0};

  static constexpr int nx = 3;
  static constexpr double scale = M_PI;
  const Kokkos::Array<double, 3> gammas = {{+1, -1, 0}};
};

TEST_F(ConductionOperatorFixture, residual_operator_zero_for_constant_data)
{
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNP1), node) = 1.0;
      *stk::mesh::field_data(q_field.field_of_state(stk::mesh::StateN), node) =
        1.0;
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNM1), node) = 1.0;
      *stk::mesh::field_data(alpha_field, node) = 1.0;
      *stk::mesh::field_data(lambda_field, node) = 1.0;
    }
  }

  auto fields = gather_required_conduction_fields<order>(meta, conn);
  ConductionResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields({{+1, -1, 0}}, fields);
  resid_op.compute(rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();

  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    ASSERT_NEAR(view_h(k, 0), 0, 1.0e-14);
  }
}

TEST_F(
  ConductionOperatorFixture, residual_operator_not_zero_for_nonconstant_data)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(1.0, 2.0);
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNP1), node) = coeff(rng);
      *stk::mesh::field_data(q_field.field_of_state(stk::mesh::StateN), node) =
        coeff(rng);
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNM1), node) = coeff(rng);
      *stk::mesh::field_data(alpha_field, node) = coeff(rng);
      *stk::mesh::field_data(lambda_field, node) = coeff(rng);
    }
  }
  stk::mesh::copy_owned_to_shared(
    mesh.get_bulk_on_host(), {&alpha_field, &lambda_field});

  auto fields = gather_required_conduction_fields<order>(meta, conn);
  ConductionResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields({{+1, -1, 0}}, fields);
  resid_op.compute(rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();
  double max_error = -1;
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    max_error = std::max(std::abs(view_h(k, 0)), max_error);
  }
  ASSERT_TRUE(max_error > 1.0e-8);
}
//
TEST_F(
  ConductionOperatorFixture,
  linearized_residual_operator_zero_for_constant_data)
{
  auto fields = gather_required_conduction_fields<order>(meta, conn);
  LinearizedResidualFields<order> coefficient_fields;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  ConductionLinearizedResidualOperator<order> cond_op(offsets, exporter);
  cond_op.set_coefficients(gammas[0], coefficient_fields);

  lhs.putScalar(0.);
  cond_op.apply(lhs, rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    ASSERT_NEAR(view_h(k, 0), 0, 1.e-14);
  }
}

TEST_F(
  ConductionOperatorFixture,
  linearized_residual_operator_not_zero_for_nonconstant_data)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(1.0, 2.0);
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(alpha_field, node) = coeff(rng);
      *stk::mesh::field_data(lambda_field, node) = coeff(rng);
    }
  }
  stk::mesh::copy_owned_to_shared(
    mesh.get_bulk_on_host(), {&alpha_field, &lambda_field});

  auto fields = gather_required_conduction_fields<order>(meta, conn);
  LinearizedResidualFields<order> coefficient_fields;
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  ConductionLinearizedResidualOperator<order> cond_op(offsets, exporter);
  cond_op.set_coefficients(gammas[0], coefficient_fields);

  lhs.randomize(-1, +1);
  lhs.sync_device();
  cond_op.apply(lhs, rhs);

  rhs.sync_host();
  auto view_h = rhs.getLocalViewHost();

  double max_error = -1;
  for (size_t k = 0u; k < rhs.getLocalLength(); ++k) {
    max_error = std::max(std::abs(view_h(k, 0)), max_error);
  }
  ASSERT_TRUE(max_error > 1.0e-8);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
