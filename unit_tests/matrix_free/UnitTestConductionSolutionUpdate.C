// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "StkConductionFixture.h"
#include "Teuchos_RCP.hpp"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionSolutionUpdate.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"

#include "gtest/gtest.h"

#include "KokkosInterface.h"
#include "Teuchos_ParameterList.hpp"

#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/CoordinateSystems.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/FieldTraits.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"
#include "stk_mesh/base/NgpForEachEntity.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_topology/topology.hpp"

#include <math.h>
#include <memory>
#include <vector>
#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace test_solution_update {
static constexpr Kokkos::Array<double, 3> gammas = {{0, 0, 0}};
}

class ConductionSolutionUpdateFixture : public ::ConductionFixture
{
protected:
  ConductionSolutionUpdateFixture()
    : ConductionFixture(nx, scale),
      linsys(
        stk::mesh::get_updated_ngp_mesh(bulk),
        meta.universal_part(),
        gid_field_ngp),
      exporter(
        Teuchos::rcpFromRef(linsys.owned_and_shared),
        Teuchos::rcpFromRef(linsys.owned)),
      offset_views(
        stk::mesh::get_updated_ngp_mesh(bulk),
        linsys.stk_lid_to_tpetra_lid,
        meta.universal_part()),
      field_update(Teuchos::ParameterList{}, linsys, exporter, offset_views)
  {
    auto& coordField =
      *meta.get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "coordinates");
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* coordptr = stk::mesh::field_data(coordField, node);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) = coordptr[0];
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateN), node) = coordptr[0];
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNM1), node) = coordptr[0];
        *stk::mesh::field_data(qtmp_field, node) = 0;
        *stk::mesh::field_data(alpha_field, node) = 1.0;
        *stk::mesh::field_data(lambda_field, node) = 1.0;
      }
    }
  }
  StkToTpetraMaps linsys;
  Tpetra::Export<> exporter;
  ConductionOffsetViews<order> offset_views;

  ConductionSolutionUpdate<order> field_update;
  LinearizedResidualFields<order> coefficient_fields;
  InteriorResidualFields<order> fields;
  static constexpr int nx = 32;
  static constexpr double scale = M_PI;
};

TEST_F(ConductionSolutionUpdateFixture, solution_state_solver_construction)
{
  ASSERT_EQ(field_update.solver().num_iterations(), 0);
}
namespace {
void
copy_tpetra_solution_vector_to_stk_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& sel,
  Kokkos::View<const typename Tpetra::Map<>::local_ordinal_type*> elid,
  typename Tpetra::MultiVector<>::dual_view_type::t_dev_const_randomread
    delta_view,
  stk::mesh::NgpField<double>& field)
{
  const int dim = delta_view.extent_int(1);
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const auto ent = mesh.get_entity(stk::topology::NODE_RANK, mi);
      const auto tpetra_lid = elid(ent.local_offset());
      for (int d = 0; d < dim; ++d) {
        field(mi, d) = delta_view(tpetra_lid, d);
      }
    });
}

} // namespace

#ifndef KOKKOS_ENABLE_GPU

TEST_F(ConductionSolutionUpdateFixture, correct_behavior_for_linear_problem)
{
  const auto conn = stk_connectivity_map<order>(mesh, meta.universal_part());
  fields = gather_required_conduction_fields<order>(meta, conn);
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  auto delta = stk::mesh::get_updated_ngp_field<double>(qtmp_field);
  field_update.compute_residual(
    test_solution_update::gammas, fields, BCDirichletFields{},
    BCFluxFields<order>{});
  auto& delta_mv = field_update.compute_delta(
    test_solution_update::gammas[0], coefficient_fields);

  copy_tpetra_solution_vector_to_stk_field(
    stk::mesh::get_updated_ngp_mesh(bulk), meta.universal_part(),
    linsys.stk_lid_to_tpetra_lid,
    delta_mv.getLocalViewDevice(Tpetra::Access::ReadOnly), delta);

  if (mesh.get_bulk_on_host().parallel_size() > 1) {
    stk::mesh::communicate_field_data<double>(
      mesh.get_bulk_on_host(), {&delta});
  }
  delta.sync_to_host();

  auto& coord_field =
    *meta.get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
      stk::topology::NODE_RANK, "coordinates");
  for (auto ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      ASSERT_NEAR(
        *stk::mesh::field_data(qtmp_field, node),
        -stk::mesh::field_data(coord_field, node)[0], 1.0e-6);
    }
  }
}

#endif // KOKKOS_ENABLE_GPU

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
