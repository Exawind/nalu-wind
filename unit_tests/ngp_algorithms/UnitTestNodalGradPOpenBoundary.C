// Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "functional"

#include "kernels/UnitTestKernelUtils.h"

#include "UnitTestHelperObjects.h"

#include "ngp_algorithms/UnitTestNgpAlgUtils.h"
#include "ngp_algorithms/NodalGradPOpenBoundaryAlg.h"
#include "ngp_algorithms/GeometryBoundaryAlg.h"
#include "ngp_algorithms/NodalGradAlgDriver.h"
#include "ngp_algorithms/GeometryAlgDriver.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

TEST_F(LowMachKernelHex8Mesh, NGP_nodal_grad_popen)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  // const std::string meshSpec = "generated:1x1x1";
  const bool doPerturb = false;
  const bool generateSidesets = true;
  const double tol = 1.0e-16;
  const double x = -0.125;
  const double y = 0.25;

  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.solutionOptions_->activateOpenMdotCorrection_ = true;

  stk::mesh::Part* surface1 = meta_->get_part("surface_1");
  {
    sierra::nalu::GeometryAlgDriver geomAlgDriver(helperObjs.realm);
    geomAlgDriver.register_face_algorithm<sierra::nalu::GeometryBoundaryAlg>(
      sierra::nalu::WALL, surface1, "geometry");
    geomAlgDriver.execute();
  }

  const auto& fieldMgr = helperObjs.realm.ngp_field_manager();
  auto ngpPres =
    fieldMgr.get_field<double>(pressure_->mesh_meta_data_ordinal());
  auto ngpDnvF =
    fieldMgr.get_field<double>(dnvField_->mesh_meta_data_ordinal());

  stk::mesh::field_fill(1.0, *pressure_);
  stk::mesh::field_fill(2.0, *dnvField_);

  ngpPres.modify_on_host();
  ngpDnvF.modify_on_host();
  ngpPres.sync_to_device();
  ngpDnvF.sync_to_device();

  const auto& bkts =
    bulk_->get_buckets(stk::topology::NODE_RANK, meta_->universal_part());

  std::function<void(bool, bool)> run_alg =
    [&](bool zeroGrad, bool useShifted) {
      stk::mesh::field_fill(0.0, *dpdx_);
      helperObjs.realm.solutionOptions_->explicitlyZeroOpenPressureGradient_ =
        zeroGrad;
      sierra::nalu::ScalarNodalGradAlgDriver algDriver(
        helperObjs.realm, pressure_->name(), "dpdx");
      algDriver
        .register_face_elem_algorithm<sierra::nalu::NodalGradPOpenBoundary>(
          sierra::nalu::OPEN, surface1, stk::topology::HEX_8,
          "nodal_grad_pressure_open_boundary", useShifted);
      algDriver.execute();

      stk::mesh::NgpField<double>& ngpdpdx =
        stk::mesh::get_updated_ngp_field<double>(*dpdx_);
      ngpdpdx.modify_on_device();
      ngpdpdx.sync_to_host();

      for (const auto* b : bkts) {
        for (const auto node : *b) {
          const double* dp = stk::mesh::field_data(*dpdx_, node);
          const double* C = stk::mesh::field_data(*coordinates_, node);
          const double re[3] = {x + y * C[0], x + y * C[1], x + y * C[2]};
          for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(re[i], dp[i], tol);
        }
      }
    };
  run_alg(false, false);
  run_alg(true, false);
  run_alg(false, true);
  run_alg(true, true);
}
