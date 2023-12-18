// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef NALU_USES_TRILINOS_SOLVERS

#include "matrix_free/MatrixFreeSolver.h"
#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionOperator.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/StkToTpetraLocalIndices.h"

#include "StkConductionFixture.h"

#include <math.h>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "BelosCGIteration.hpp"
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosMultiVecTraits.hpp"
#include "BelosOperatorTraits.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include "BelosStatusTestGenResNorm.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosTypes.hpp"
#include "Kokkos_Core.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace test_belos_solver {
static constexpr Kokkos::Array<double, 3> gammas{{+1, -1, 0}};
}

class SolverFixture : public ::ConductionFixture
{
protected:
  static constexpr int nx = 32;
  static constexpr double scale = M_PI;

  SolverFixture()
    : ConductionFixture(nx, scale),
      owned_map(make_owned_row_map(mesh, meta.universal_part())),
      owned_and_shared_map(make_owned_and_shared_row_map(
        mesh, meta.universal_part(), gid_field_ngp)),
      exporter(
        Teuchos::rcpFromRef(owned_and_shared_map),
        Teuchos::rcpFromRef(owned_map)),
      elid(make_stk_lid_to_tpetra_lid_map(
        mesh,
        meta.universal_part(),
        gid_field_ngp,
        owned_and_shared_map.getLocalMap())),
      conn(stk_connectivity_map<order>(mesh, meta.universal_part())),
      offsets(create_offset_map<order>(mesh, meta.universal_part(), elid)),
      resid_op(offsets, exporter),
      lin_op(offsets, exporter)
  {
    auto& coord_field = coordinate_field();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* coordptr = stk::mesh::field_data(coord_field, node);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) =
          std::cos(coordptr[0]);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateN), node) =
          std::cos(coordptr[0]);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNM1), node) =
          std::cos(coordptr[0]);
        *stk::mesh::field_data(qtmp_field, node) = 0;
        *stk::mesh::field_data(alpha_field, node) = 1.0;
        *stk::mesh::field_data(lambda_field, node) = 1.0;
      }
    }
    fields = gather_required_conduction_fields<order>(meta, conn);
    coefficient_fields.volume_metric = fields.volume_metric;
    coefficient_fields.diffusion_metric = fields.diffusion_metric;
  }

  const Tpetra::Map<> owned_map;
  const Tpetra::Map<> owned_and_shared_map;
  const Tpetra::Export<> exporter;
  const const_entity_row_view_type elid;

  const elem_mesh_index_view<order> conn;
  const elem_offset_view<order> offsets;
  const node_offset_view dirichlet_bc_offsets{"empty_dirichlet", 0};

  ConductionResidualOperator<order> resid_op;
  ConductionLinearizedResidualOperator<order> lin_op;

  InteriorResidualFields<order> fields;
  LinearizedResidualFields<order> coefficient_fields;
};

TEST_F(SolverFixture, solve_zero_rhs)
{
  auto list = Teuchos::ParameterList{};
  MatrixFreeSolver solver(lin_op, 1, list);
  lin_op.set_coefficients(test_belos_solver::gammas[0], coefficient_fields);

  solver.rhs().putScalar(0.);
  solver.solve();
  ASSERT_EQ(solver.num_iterations(), 0);
}

TEST_F(SolverFixture, solve_harmonic)
{
  auto list = Teuchos::ParameterList{};
  MatrixFreeSolver solver(lin_op, 1, list);
  resid_op.set_fields(test_belos_solver::gammas, fields);
  resid_op.compute(solver.rhs());
  lin_op.set_coefficients(test_belos_solver::gammas[0], coefficient_fields);
  solver.solve();
  ASSERT_TRUE(solver.num_iterations() > 1 && solver.num_iterations() < 1000);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif // NALU_USES_TRILINOS_SOLVERS
