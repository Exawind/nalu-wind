// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionGatheredFieldManager.h"

#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionInfo.h"
#include "StkConductionFixture.h"

#include <KokkosInterface.h>
#include "Kokkos_Macros.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/NgpForEachEntity.hpp"
#include "stk_simd/Simd.hpp"

#include <memory>

namespace sierra {
namespace nalu {
namespace matrix_free {

class GatheredFieldManagerFixture : public ::ConductionFixture
{
protected:
  static constexpr int nx = 32;
  static constexpr double scale = M_PI;
  GatheredFieldManagerFixture()
    : ConductionFixture(nx, scale),
      field_gather(bulk, meta.universal_part(), {})
  {
    auto& coordField =
      *meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
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
  ConductionGatheredFieldManager<order> field_gather;
};

TEST_F(GatheredFieldManagerFixture, gather_all)
{
  field_gather.gather_all();
  auto residual_fields = field_gather.get_residual_fields();
  EXPECT_TRUE(residual_fields.volume_metric.extent_int(0) > 1);
  EXPECT_TRUE(residual_fields.qm1.extent_int(0) > 1);
  EXPECT_TRUE(residual_fields.qp0.extent_int(0) > 1);
  EXPECT_TRUE(residual_fields.qp1.extent_int(0) > 1);
}

TEST_F(GatheredFieldManagerFixture, swap_states)
{
  field_gather.gather_all();
  auto residual_fields = field_gather.get_residual_fields();
  auto qm1_label = residual_fields.qm1.label();
  auto qp0_label = residual_fields.qp0.label();
  auto qp1_label = residual_fields.qp1.label();

  field_gather.swap_states();
  residual_fields = field_gather.get_residual_fields();
  EXPECT_EQ(residual_fields.qm1.label(), qp0_label);
  EXPECT_EQ(residual_fields.qp0.label(), qp1_label);
  EXPECT_EQ(residual_fields.qp1.label(), qm1_label);
}

namespace {
template <int p>
double
sum_field(scalar_view<p> qp1)
{
  double sum_prev = 0;
  Kokkos::parallel_reduce(
    sierra::nalu::DeviceRangePolicy(0, qp1.extent_int(0)),
    KOKKOS_LAMBDA(int index, double& sumval) {
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            for (int n = 0; n < simd_len; ++n) {
              sumval += stk::simd::get_data(qp1(index, k, j, i), n);
            }
          }
        }
      }
    },
    sum_prev);
  return sum_prev;
}

void
double_field(
  const stk::mesh::NgpMesh& mesh,
  stk::mesh::Selector selector,
  stk::mesh::NgpField<double> field)
{
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, selector,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) { field.get(mi, 0) *= 2; });
}

} // namespace

TEST_F(GatheredFieldManagerFixture, update_solution)
{
  auto sol_field = stk::mesh::get_updated_ngp_field<double>(q_field);
  sol_field.set_all(mesh, 1.);

  field_gather.gather_all();

  auto sum_prev = sum_field<order>(field_gather.get_residual_fields().qp1);

  double_field(mesh, meta.universal_part(), sol_field);
  field_gather.update_solution_fields();
  auto sum_post = sum_field<order>(field_gather.get_residual_fields().qp1);

  EXPECT_DOUBLE_EQ(sum_prev, sum_post / 2);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
