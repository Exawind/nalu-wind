// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "AlgTraits.h"
#include "ngp_algorithms/SDRWallFuncAlg.h"
#include "ngp_algorithms/SDRLowReWallAlg.h"
#include "ngp_algorithms/SDRWallFuncAlgDriver.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "utils/StkHelpers.h"

TEST_F(SSTKernelHex8Mesh, NGP_sdr_wall_lowRE)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  solnOpts_.initialize_turbulence_constants();

  // Set necessary fields
  const double viscosity = 1.0e-5;
  const double rho = 1.0;
  stk::mesh::field_fill(rho, *density_);
  stk::mesh::field_fill(viscosity, *visc_);

  const bool useShifted = false;
  unit_test_utils::HelperObjects helperObjs(
    *bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.solutionOptions_->initialize_turbulence_constants();

  auto* part = meta_->get_part("surface_5");
  auto* surfPart = part->subsets()[0];
  sierra::nalu::SDRWallFuncAlgDriver algDriver(helperObjs.realm);
  algDriver.register_face_elem_algorithm<sierra::nalu::SDRLowReWallAlg>(
    sierra::nalu::WALL, surfPart,
    sierra::nalu::get_elem_topo(helperObjs.realm, *surfPart),
    "sdr_lowre_wall", useShifted);

  algDriver.execute();

  {
    const double tol = 1.0e-16;

    // Wall distance and area check
    const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
    auto& ngpSdrBC = fieldMgr.get_field<double>(
      sdrWallbc_->mesh_meta_data_ordinal());
    auto& ngpWallArea = fieldMgr.get_field<double>(
      sdrWallArea_->mesh_meta_data_ordinal());

    ngpSdrBC.sync_to_host();
    ngpWallArea.sync_to_host();

    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    const double wAreaGold = 0.25;

    auto& realm = helperObjs.realm;
    const double ypSqr = 0.25 * 0.25;
    const double wallFactor = realm.get_turb_model_constant(sierra::nalu::TM_SDRWallFactor);
    const double betaOne = realm.get_turb_model_constant(sierra::nalu::TM_betaOne);
    const double nu = viscosity / rho;
    const double sdrGold = wallFactor * 6.0 * nu / (betaOne * ypSqr);

    for (const auto* b: bkts)
      for (const auto& node: *b) {
        const double* sdrVal = stk::mesh::field_data(*sdrWallbc_, node);
        const double* areaVal = stk::mesh::field_data(*sdrWallArea_, node);
        EXPECT_NEAR(sdrVal[0], sdrGold, tol);
        EXPECT_NEAR(areaVal[0], wAreaGold, tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_sdr_wall_func)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  solnOpts_.initialize_turbulence_constants();

  // Set necessary fields
  const double utau = 0.5;
  stk::mesh::field_fill(utau, *wallFricVel_);

  unit_test_utils::HelperObjects helperObjs(
    *bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.solutionOptions_->initialize_turbulence_constants();

  auto* part = meta_->get_part("surface_5");
  auto* surfPart = part->subsets()[0];
  sierra::nalu::SDRWallFuncAlgDriver algDriver(helperObjs.realm);
  algDriver.register_face_elem_algorithm<sierra::nalu::SDRWallFuncAlg>(
    sierra::nalu::WALL, surfPart,
    sierra::nalu::get_elem_topo(helperObjs.realm, *surfPart),
    "sdr_lowre_wall");

  algDriver.execute();

  {
    const double tol = 1.0e-16;

    // Wall distance and area check
    const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
    auto& ngpSdrBC = fieldMgr.get_field<double>(
      sdrWallbc_->mesh_meta_data_ordinal());
    auto& ngpWallArea = fieldMgr.get_field<double>(
      sdrWallArea_->mesh_meta_data_ordinal());

    ngpSdrBC.sync_to_host();
    ngpWallArea.sync_to_host();

    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    const double wAreaGold = 0.25;

    auto& realm = helperObjs.realm;
    const double yplus = 0.25;
    const double kappa = realm.get_turb_model_constant(sierra::nalu::TM_kappa);
    const double sqrtBetaStar = std::sqrt(realm.get_turb_model_constant(sierra::nalu::TM_betaStar));
    const double sdrGold = utau / (sqrtBetaStar * kappa * yplus);

    for (const auto* b: bkts)
      for (const auto& node: *b) {
        const double* sdrVal = stk::mesh::field_data(*sdrWallbc_, node);
        const double* areaVal = stk::mesh::field_data(*sdrWallArea_, node);
        EXPECT_NEAR(sdrVal[0], sdrGold, tol);
        EXPECT_NEAR(areaVal[0], wAreaGold, tol);
      }
  }
}
