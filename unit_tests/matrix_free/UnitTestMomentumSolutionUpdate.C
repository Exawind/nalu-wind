// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/MomentumSolutionUpdate.h"
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

namespace test_momentum_solution_update {
static constexpr double time_scale = 1;
}

class MomentumSolutionUpdateFixture : public LowMachFixture
{
protected:
  MomentumSolutionUpdateFixture()
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
    const auto conn =
      stk_connectivity_map<order>(mesh(), meta.universal_part());
    fields = gather_required_lowmach_fields<order>(meta, conn);
    coefficient_fields.unscaled_volume_metric = fields.unscaled_volume_metric;
    coefficient_fields.volume_metric = fields.volume_metric;
    coefficient_fields.diffusion_metric = fields.diffusion_metric;
    coefficient_fields.laplacian_metric = fields.laplacian_metric;
    coefficient_fields.advection_metric = fields.advection_metric;
  }

  StkToTpetraMaps linsys;
  Tpetra::Export<> exporter;
  elem_mesh_index_view<order> conn;
  elem_offset_view<order> offsets;
  MomentumSolutionUpdate<order> field_update;
  LowMachLinearizedResidualFields<order> coefficient_fields;
  LowMachResidualFields<order> fields;
  LowMachBCFields<order> bc;

  static constexpr int nx = 8;
  static constexpr double scale = M_PI;
};

TEST_F(MomentumSolutionUpdateFixture, solution_state_solver_construction)
{
  ASSERT_EQ(field_update.solver().num_iterations(), 0);
}

TEST_F(MomentumSolutionUpdateFixture, solve_is_reasonable)
{
  const double inv_dt = 1. / test_momentum_solution_update::time_scale;
  field_update.compute_residual({{inv_dt, -inv_dt, 0}}, fields, bc);
  field_update.compute_delta(inv_dt, coefficient_fields);

  ASSERT_TRUE(
    field_update.num_iterations() > 2 && field_update.num_iterations() < 30);
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
