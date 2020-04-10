#include "matrix_free/ConductionSolutionUpdate.h"

#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionInterior.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/StkToTpetraMap.h"
#include "StkConductionFixture.h"

#include "Teuchos_RCP.hpp"
#include "Tpetra_Export.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"

#include <memory>

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace test_solution_update {
static constexpr Kokkos::Array<double, 3> gammas = {{0, 0, 0}};
}

class SolutionUpdateFixture : public ::ConductionFixture
{
protected:
  SolutionUpdateFixture()
    : ConductionFixture(nx, scale),
      field_update(
        Teuchos::ParameterList{}, mesh, gid_field_ngp, meta.universal_part())
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
  ConductionSolutionUpdate<order> field_update;
  LinearizedResidualFields<order> coefficient_fields;
  InteriorResidualFields<order> fields;
  static constexpr int nx = 32;
  static constexpr double scale = M_PI;
};

TEST_F(SolutionUpdateFixture, solution_state_solver_construction)
{
  ASSERT_EQ(field_update.solver().num_iterations(), 0);
}

TEST_F(SolutionUpdateFixture, correct_behavior_for_linear_problem)
{
  const auto conn = stk_connectivity_map<order>(mesh, meta.universal_part());
  fields = gather_required_conduction_fields<order>(meta, conn);
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  auto delta = stk::mesh::get_updated_ngp_field<double>(qtmp_field);
  field_update.compute_residual(
    test_solution_update::gammas, fields, BCDirichletFields{},
    BCFluxFields<order>{});
  field_update.compute_delta(
    test_solution_update::gammas[0], coefficient_fields, delta);

  if (mesh.get_bulk_on_host().parallel_size() > 1) {
    stk::mesh::communicate_field_data<double>(
      mesh.get_bulk_on_host(), {&delta});
  }
  delta.sync_to_host();
  delta.sync_to_device();

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

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
