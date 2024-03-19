// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ContinuityOperator.h"

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

class ContinuityOperatorFixture : public LowMachFixture
{
protected:
  static constexpr int nodes_per_elem = (order + 1) * (order + 1) * (order + 1);

  ContinuityOperatorFixture()
    : LowMachFixture(nx, scale),
      owned_map(make_owned_row_map(mesh(), active())),
      owned_and_shared_map(
        make_owned_and_shared_row_map(mesh(), active(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      lhs(Teuchos::rcpFromRef(owned_map), 1),
      rhs(Teuchos::rcpFromRef(owned_map), 1),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh(), active(), gid_field_ngp, owned_and_shared_map.getLocalMap())),
      elid_h(Kokkos::create_mirror_view_and_copy(
        Kokkos::DefaultHostExecutionSpace{}, elid)),
      conn(stk_connectivity_map<order>(mesh(), active())),
      offsets(create_offset_map<order>(mesh(), active(), elid))
  {
    lhs.putScalar(0.);
    rhs.putScalar(0.);
  }

  const_scs_scalar_view<order> compute_mdot(double tau)
  {
    auto fields = gather_required_lowmach_fields<order>(meta, conn);
    geom::linear_advection_metric<order>(
      tau, fields.area_metric, fields.laplacian_metric, fields.rho, fields.up1,
      fields.gp, fields.pressure, fields.advection_metric);
    return fields.advection_metric;
  }

  const_scs_vector_view<order> compute_metric()
  {
    vector_view<order> coordinates("coordinates", offsets.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(coordinate_field()),
      coordinates);

    return geom::diffusion_metric<order>(coordinates);
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  Tpetra::Export<> exporter;
  Tpetra::MultiVector<> lhs;
  Tpetra::MultiVector<> rhs;
  const const_entity_row_view_type elid;

  using host_space = Kokkos::DefaultHostExecutionSpace;
  using host_type =
    decltype(Kokkos::create_mirror_view_and_copy(host_space{}, elid));
  const host_type elid_h;

  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;

  static constexpr int nx = 4;
  static constexpr double scale = 1;
};

TEST_F(
  ContinuityOperatorFixture, residual_operator_zero_in_interior_for_free_flow)
{
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      stk::mesh::field_data(velocity_field, node)[0] = 1.0;
      stk::mesh::field_data(velocity_field, node)[1] = 0.0;
      stk::mesh::field_data(velocity_field, node)[2] = 0.0;
      *stk::mesh::field_data(density_field, node) = 1.0;
    }
  }

  const double tau = 1;
  const auto mdot = compute_mdot(tau);
  ContinuityResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_fields(tau, mdot);
  resid_op.compute(rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  //  side should be #(faces connectded to node) * (scale/nx)^2
  const auto interior_selector =
    stk::mesh::Selector(meta.universal_part()) - stk::mesh::Selector(side());
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, interior_selector)) {
    for (auto node : *ib) {
      const int lid = elid_h(node.local_offset());
      if (lid < view_h.extent_int(0)) {
        ASSERT_NEAR(view_h(lid, 0), 0, 1.0e-14);
      }
    }
  }
}

TEST_F(
  ContinuityOperatorFixture,
  linearized_residual_operator_zero_for_linear_function)
{
  {
    auto host_lhs = lhs.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto x = *stk::mesh::field_data(coordinate_field(), node);
        const int lid = elid_h(node.local_offset());
        if (lid < host_lhs.extent_int(0)) {
          host_lhs(lid, 0) = x;
        }
      }
    }
  }

  const auto metric = compute_metric();
  ContinuityLinearizedResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_metric(metric);
  resid_op.apply(lhs, rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);
  const auto interior_selector = active() - side();
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, interior_selector)) {
    for (auto node : *ib) {
      const int lid = elid_h(node.local_offset());
      if (lid < view_h.extent_int(0)) {
        ASSERT_NEAR(view_h(lid, 0), 0, 1.0e-14);
      }
    }
  }
}

TEST_F(
  ContinuityOperatorFixture,
  linearized_residual_operator_nonzero_for_quadratic_function)
{
  {
    auto host_lhs = lhs.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (const auto* ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto x = *stk::mesh::field_data(coordinate_field(), node);
        const int lid = elid_h(node.local_offset());
        if (lid < host_lhs.extent_int(0)) {
          host_lhs(lid, 0) = x * x;
        }
      }
    }
  }

  const auto metric = compute_metric();
  ContinuityLinearizedResidualOperator<order> resid_op(offsets, exporter);
  resid_op.set_metric(metric);
  resid_op.apply(lhs, rhs);

  auto view_h = rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  //  side should be #(faces connectded to node) * (scale/nx)^2
  double max_val = -1;
  const auto interior_selector = active() - side();
  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, interior_selector)) {
    for (auto node : *ib) {
      const int lid = elid_h(node.local_offset());
      if (lid < view_h.extent_int(0)) {
        max_val = stk::math::max(stk::math::abs(view_h(lid, 0)), max_val);
      }
    }
  }
  ASSERT_GT(max_val, 1.0e-2 * scale * scale / (nx * nx));
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
