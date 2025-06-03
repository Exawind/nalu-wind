// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "StkConductionFixture.h"

#include "matrix_free/ConductionFields.h"
#include "matrix_free/ConductionInfo.h"
#include "matrix_free/ConductionUpdate.h"
#include "matrix_free/EquationUpdate.h"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/NgpForEachEntity.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_topology/topology.hpp"

#include <math.h>
#include <algorithm>
#include <limits>
#include <memory>

namespace sierra {
namespace nalu {
namespace matrix_free {
namespace test_simulation {

constexpr int nx = 40;
constexpr double scale = M_PI;

constexpr double rhocp = 1000;
constexpr double kappa = 4.5e-2;

constexpr double smallish = 1.0e-4;

constexpr int kx = 2;
constexpr int ky = 3;
constexpr int kz = 4;

void
field_add(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  const stk::mesh::NgpField<double>& x,
  stk::mesh::NgpField<double>& y)
{
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, active,
    KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& mi) {
      y.get(mi, 0) += x.get(mi, 0);
    });
  y.modify_on_device();
}

stk::mesh::NgpField<double>&
get_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  STK_ThrowAssert(meta.get_field(stk::topology::NODE_RANK, name));
  STK_ThrowAssert(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return stk::mesh::get_updated_ngp_field<double>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
}

double
initial_condition(double x, double y, double z)
{
  constexpr double xshift = (kx % 2 == 1) ? M_PI_2 : 0;
  constexpr double yshift = (ky % 2 == 1) ? M_PI_2 : 0;
  constexpr double zshift = (kz % 2 == 1) ? M_PI_2 : 0;
  return std::cos(kx * x + xshift) * std::cos(ky * y + yshift) *
         std::cos(kz * z + zshift);
}

double
solution(double t, double x, double y, double z)
{
  return std::exp(-(kx * kx + ky * ky + kz * kz) * kappa / rhocp * t) *
         initial_condition(x, y, z);
}

class ConductionSimulationFixture : public ::ConductionFixture
{
protected:
  ConductionSimulationFixture()
    : ConductionFixture(nx, scale),
      update(
        make_updater<ConductionUpdate>(
          order,
          bulk,
          Teuchos::ParameterList{},
          meta.universal_part(),
          stk::mesh::Selector{},
          stk::mesh::Selector{}))
  {
    auto& coordField = coordinate_field();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const double x = stk::mesh::field_data(coordField, node)[0];
        const double y = stk::mesh::field_data(coordField, node)[1];
        const double z = stk::mesh::field_data(coordField, node)[2];

        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) =
          initial_condition(x, y, z);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateN), node) =
          initial_condition(x, y, z);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNM1), node) =
          initial_condition(x, y, z);

        *stk::mesh::field_data(qtmp_field, node) = 0;
        *stk::mesh::field_data(alpha_field, node) = rhocp;
        *stk::mesh::field_data(lambda_field, node) = kappa;
      }
    }
  }

  double run_simulation(double dt, double final_time)
  {
    double time = 0;
    Kokkos::Array<double, 3> gammas = {{+1 / dt, -1 / dt, 0}};
    update->initialize();
    update->compute_preconditioner(gammas[0]);
    update->compute_update(
      gammas, get_ngp_field(meta, conduction_info::q_name));
    update->update_solution_fields();
    time += dt;

    gammas = {{1.5 / dt, -2 / dt, 0.5 / dt}};
    update->compute_preconditioner(gammas[0]);
    while (time < final_time - dt / 2) {
      bulk.update_field_data_states();
      get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNM1)
        .swap(get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateN));
      get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateN)
        .swap(
          get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNP1));
      update->swap_states();
      update->predict_state();

      update->compute_update(
        gammas, get_ngp_field(meta, conduction_info::q_name));
      update->update_solution_fields();
      time += dt;
    }
    get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNP1)
      .sync_to_host();
    return time;
  }

  double max_value()
  {
    auto& qp1 = q_field.field_of_state(stk::mesh::StateNP1);
    double max_val = std::numeric_limits<double>::lowest();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        max_val =
          std::max(std::abs(*stk::mesh::field_data(qp1, node)), max_val);
      }
    }
    return max_val;
  }

  double error(double time)
  {
    auto& coords = coordinate_field();
    auto& qp1 = q_field.field_of_state(stk::mesh::StateNP1);
    double max_error = std::numeric_limits<double>::lowest();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const double x = stk::mesh::field_data(coords, node)[0];
        const double y = stk::mesh::field_data(coords, node)[1];
        const double z = stk::mesh::field_data(coords, node)[2];
        const double val = *stk::mesh::field_data(qp1, node);
        max_error = std::max(max_error, val - solution(time, x, y, z));
      }
    }
    return max_error;
  }

  std::unique_ptr<EquationUpdate> update;
};

TEST_F(ConductionSimulationFixture, heat_conduction_reduces_peak_value)
{
  auto max_val_pre = max_value();
  run_simulation(5e-3, 1.5e-2);
  auto max_val_post = max_value();
  ASSERT_GT(max_val_pre, max_val_post);
}

TEST_F(ConductionSimulationFixture, difference_from_exact_solution_is_smallish)
{
  const double dt = 5.0e-5 * rhocp / kappa;
  const double final_time = run_simulation(dt, 3 * dt);
  ASSERT_LT(error(final_time), smallish);
}

class QuadraticElementConductionSimulationFixture : public ::ConductionFixtureP2
{
protected:
  QuadraticElementConductionSimulationFixture()
    : ConductionFixtureP2(nx / order, scale),
      update(
        make_updater<ConductionUpdate>(
          order,
          bulk,
          Teuchos::ParameterList{},
          meta.universal_part(),
          stk::mesh::Selector{},
          stk::mesh::Selector{}))
  {
    auto& coordField = coordinate_field();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const double x = stk::mesh::field_data(coordField, node)[0];
        const double y = stk::mesh::field_data(coordField, node)[1];
        const double z = stk::mesh::field_data(coordField, node)[2];
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNP1), node) =
          initial_condition(x, y, z);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateN), node) =
          initial_condition(x, y, z);
        *stk::mesh::field_data(
          q_field.field_of_state(stk::mesh::StateNM1), node) =
          initial_condition(x, y, z);
        *stk::mesh::field_data(qtmp_field, node) = 0;
        *stk::mesh::field_data(alpha_field, node) = rhocp;
        *stk::mesh::field_data(lambda_field, node) = kappa;
      }
    }
  }

  double max_value()
  {
    auto& qp1 = q_field.field_of_state(stk::mesh::StateNP1);
    double max_val = std::numeric_limits<double>::lowest();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        max_val =
          std::max(std::abs(*stk::mesh::field_data(qp1, node)), max_val);
      }
    }
    return max_val;
  }

  double error(double time)
  {
    auto& coords = coordinate_field();
    auto& qp1 = q_field.field_of_state(stk::mesh::StateNP1);
    double max_error = std::numeric_limits<double>::lowest();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const double x = stk::mesh::field_data(coords, node)[0];
        const double y = stk::mesh::field_data(coords, node)[1];
        const double z = stk::mesh::field_data(coords, node)[2];
        const double val = *stk::mesh::field_data(qp1, node);
        max_error = std::max(max_error, val - solution(time, x, y, z));
      }
    }
    return max_error;
  }

  double run_simulation(double dt, double final_time)
  {
    double time = 0;
    Kokkos::Array<double, 3> gammas = {{+1 / dt, -1 / dt, 0}};
    update->initialize();
    update->compute_preconditioner(gammas[0]);
    update->compute_update(
      gammas, get_ngp_field(meta, conduction_info::q_name));
    update->update_solution_fields();
    time += dt;

    gammas = {{1.5 / dt, -2 / dt, 0.5 / dt}};
    update->compute_preconditioner(gammas[0]);
    while (time < final_time - dt / 2) {
      bulk.update_field_data_states();
      get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNM1)
        .swap(get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateN));
      get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateN)
        .swap(
          get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNP1));
      update->swap_states();
      update->predict_state();

      update->compute_update(
        gammas, get_ngp_field(meta, conduction_info::q_name));
      update->update_solution_fields();
      time += dt;
    }
    get_ngp_field(meta, conduction_info::q_name, stk::mesh::StateNP1)
      .sync_to_host();
    return time;
  }

  std::unique_ptr<EquationUpdate> update;
};

TEST_F(
  QuadraticElementConductionSimulationFixture,
  heat_conduction_reduces_peak_value)
{
  auto max_val_pre = max_value();
  run_simulation(5e-3, 1.5e-2);
  auto max_val_post = max_value();
  ASSERT_GT(max_val_pre, max_val_post);
}

TEST_F(
  QuadraticElementConductionSimulationFixture,
  difference_from_exact_solution_is_smallish)
{
  const double dt = 5.0e-5 * rhocp / kappa;
  const double final_time = run_simulation(dt, 6 * dt);
  ASSERT_LT(error(final_time), smallish);
}

} // namespace test_simulation
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
