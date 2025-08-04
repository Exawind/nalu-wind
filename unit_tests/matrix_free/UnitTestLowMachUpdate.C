// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "StkLowMachFixture.h"

#include "matrix_free/LowMachFields.h"
#include "matrix_free/LowMachInfo.h"
#include "matrix_free/LowMachUpdate.h"
#include "matrix_free/EquationUpdate.h"

#include "UnitTestUtils.h"

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "matrix_free/StkToTpetraMap.h"

#include <math.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <stk_mesh/base/GetNgpField.hpp>

namespace sierra {
namespace nalu {
namespace matrix_free {

constexpr int nx = 16;
constexpr double scale = M_PI;

class LowMachSimulationFixture : public LowMachFixture
{
protected:
  LowMachSimulationFixture()
    : LowMachFixture(nx, scale),
      linsys(mesh(), active(), gid_field_ngp),
      update(
        make_updater<LowMachUpdate>(
          order,
          bulk,
          Teuchos::ParameterList{},
          Teuchos::ParameterList{},
          Teuchos::ParameterList{},
          active(),
          stk::mesh::Selector{},
          linsys.owned,
          linsys.owned_and_shared,
          linsys.stk_lid_to_tpetra_lid))
  {
  }

  double max_value()
  {
    double max_val = std::numeric_limits<double>::lowest();
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        for (int d = 0; d < 3; ++d) {
          max_val = std::max(
            std::abs(stk::mesh::field_data(velocity_field, node)[d]), max_val);
        }
      }
    }
    return max_val;
  }
  StkToTpetraMaps linsys;
  std::unique_ptr<LowMachEquationUpdate> update;
};

namespace {
void
copy_field(
  const stk::mesh::NgpMesh& mesh,
  const stk::mesh::Selector& active,
  stk::mesh::NgpField<double> dst,
  stk::mesh::NgpField<double> src)
{
  src.sync_to_device();
  stk::mesh::for_each_entity_run(
    mesh, stk::topology::NODE_RANK, active,
    KOKKOS_LAMBDA(stk::mesh::FastMeshIndex mi) {
      const int dim = dst.get_num_components_per_entity(mi);
      for (int d = 0; d < dim; ++d) {
        dst.get(mi, d) = src.get(mi, d);
      };
    });
  dst.modify_on_device();
}
} // namespace

TEST_F(LowMachSimulationFixture, reduce_peak_velocity)
{
  auto rho = stk::mesh::get_updated_ngp_field<double>(density_field);
  auto vel = stk::mesh::get_updated_ngp_field<double>(velocity_field);
  auto press = stk::mesh::get_updated_ngp_field<double>(pressure_field);
  auto dpdx = stk::mesh::get_updated_ngp_field<double>(dpdx_field);
  auto dpdx_tmp = stk::mesh::get_updated_ngp_field<double>(dpdx_tmp_field);

  Kokkos::Array<double, 3> gammas{{100, -100, 0}};
  auto max_val_pre = max_value();

  update->initialize();

  update->gather_velocity();
  update->gather_pressure();
  update->gather_grad_p();
  update->update_advection_metric(0);

  update->update_provisional_velocity(gammas, vel);
  update->gather_velocity();

  update->update_pressure(1. / gammas[0], press);
  update->gather_pressure();

  copy_field(mesh(), active(), dpdx_tmp, dpdx);
  update->update_pressure_gradient(dpdx);

  update->gather_grad_p();
  update->project_velocity(1. / gammas[0], rho, dpdx_tmp, dpdx, vel);

  vel.sync_to_host();

  const bool doOutput = false;
  if (doOutput) {
    unit_test_utils::dump_mesh(bulk, {&velocity_field});
  }
  auto max_val_post = max_value();
  ASSERT_GT(max_val_pre, max_val_post);
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
