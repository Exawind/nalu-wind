// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ContinuitySolutionUpdate.h"
#include "matrix_free/LowMachFields.h"
#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"

#include "StkLowMachFixture.h"
#include "Teuchos_RCP.hpp"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Tpetra_Export.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Entity.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
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

namespace test_continuity_solution_update {
static constexpr double time_scale = 1;
}

class ContinuitySolutionUpdateFixture : public LowMachFixture
{
protected:
  ContinuitySolutionUpdateFixture()
    : LowMachFixture(nx, scale),
      linsys(
        stk::mesh::get_updated_ngp_mesh(bulk),
        meta.universal_part(),
        gid_field_ngp),
      exporter(
        Teuchos::rcpFromRef(linsys.owned_and_shared),
        Teuchos::rcpFromRef(linsys.owned)),
      conn(stk_connectivity_map<order>(mesh(), meta.universal_part())),
      offsets(create_offset_map<order>(
        mesh(), meta.universal_part(), linsys.stk_lid_to_tpetra_lid)),
      field_update(Teuchos::ParameterList{}, linsys, exporter, offsets)
  {
  }

  StkToTpetraMaps linsys;
  Tpetra::Export<> exporter;
  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;
  ContinuitySolutionUpdate<order> field_update;
  LowMachLinearizedResidualFields<order> coefficient_fields;
  LowMachResidualFields<order> fields;
  static constexpr int nx = 8;
  static constexpr double scale = M_PI;
};

TEST_F(ContinuitySolutionUpdateFixture, solution_state_solver_construction)
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

TEST_F(ContinuitySolutionUpdateFixture, solve_is_reasonable)
{
  const auto conn = stk_connectivity_map<order>(mesh(), meta.universal_part());
  fields = gather_required_lowmach_fields<order>(meta, conn);
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  auto delta = stk::mesh::get_updated_ngp_field<double>(pressure_field);
  // use a tmp
  const double dt = test_continuity_solution_update::time_scale;
  field_update.compute_residual(dt, fields.advection_metric);
  auto& delta_mv = field_update.compute_delta(fields.laplacian_metric);

  copy_tpetra_solution_vector_to_stk_field(
    stk::mesh::get_updated_ngp_mesh(bulk), meta.universal_part(),
    linsys.stk_lid_to_tpetra_lid,
    delta_mv.getLocalViewDevice(Tpetra::Access::ReadOnly), delta);

  if (bulk.parallel_size() > 1) {
    stk::mesh::communicate_field_data<double>(bulk, {&delta});
  }
  delta.sync_to_host();

  ASSERT_TRUE(
    field_update.num_iterations() > 1 && field_update.num_iterations() < 100);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
