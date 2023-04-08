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
#include "ngp_algorithms/MdotEdgeAlg.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_algorithms/MdotDensityAccumAlg.h"
#include "ngp_algorithms/MdotOpenCorrectorAlg.h"
#include "ngp_algorithms/MdotInflowAlg.h"
#include "ngp_algorithms/MdotOpenEdgeAlg.h"

TEST_F(MomentumEdgeHex8Mesh, NGP_mdot_calc_edge)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.mdotInterpRhoUTogether_ = true;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.interiorPartVec_.push_back(partVec_[0]);

  stk::mesh::field_fill(1.0, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  stk::mesh::field_fill(10.0, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  stk::mesh::field_fill(0.0, *pressure_);
  pressure_->modify_on_host();
  pressure_->sync_to_device();

  stk::mesh::field_fill(0.0, *dpdx_);
  dpdx_->modify_on_host();
  dpdx_->sync_to_device();

  stk::mesh::field_fill(0.0, *massFlowRateEdge_);
  massFlowRateEdge_->modify_on_host();
  massFlowRateEdge_->sync_to_device();

  sierra::nalu::MdotEdgeAlg mdotAlg(helperObjs.realm, partVec_[0]);
  mdotAlg.execute();

  const auto& fieldMgr = helperObjs.realm.ngp_field_manager();
  auto ngpMdot =
    fieldMgr.get_field<double>(massFlowRateEdge_->mesh_meta_data_ordinal());
  ngpMdot.modify_on_device();
  ngpMdot.sync_to_host();

  {
    const double tol = 1.0e-14;
    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::EDGE_RANK, sel);

    for (const auto* b : bkts)
      for (const auto edge : *b) {
        const double* mdot = stk::mesh::field_data(*massFlowRateEdge_, edge);
        EXPECT_NEAR(mdot[0], 2.5, tol);
      }
  }
}

TEST_F(MomentumEdgeHex8Mesh, NGP_mdot_rho_accum)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  fill_mesh_and_init_fields();

  // Set up time integrator
  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.secondOrderTimeAccurate_ = false;
  timeIntegrator.timeStepN_ = 0.125;
  timeIntegrator.timeStepNm1_ = 0.125;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  stk::mesh::field_fill(2.0, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  stk::mesh::field_fill(1.0, density_->field_of_state(stk::mesh::StateN));
  density_->field_of_state(stk::mesh::StateN).modify_on_host();
  density_->field_of_state(stk::mesh::StateN).sync_to_device();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.interiorPartVec_.push_back(partVec_[0]);
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;
  helperObjs.realm.solutionOptions_->mdotInterpRhoUTogether_ = true;

  // Instantiate the algorithm driver
  const bool elementContinuityEqs = true;
  const bool lumpedMass = true;
  sierra::nalu::MdotAlgDriver mdotDriver(
    helperObjs.realm, elementContinuityEqs);

  mdotDriver.register_elem_algorithm<sierra::nalu::MdotDensityAccumAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "mdot_rho_acc", mdotDriver,
    lumpedMass);

  mdotDriver.execute();

  const double expectedValue =
    1.0 * sierra::nalu::AlgTraitsHex8::nodesPerElement_;
  EXPECT_NEAR(mdotDriver.mdot_rho_accum(), expectedValue, 1.0e-15);
}

TEST_F(MomentumEdgeHex8Mesh, NGP_mdot_open_correction)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  stk::mesh::field_fill(0.0, *openMassFlowRate_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.solutionOptions_->activateOpenMdotCorrection_ = true;
  auto* part = meta_->get_part("surface_5");
  auto* surfPart = part->subsets()[0];
  const bool elementContinuityEqs = true;

  sierra::nalu::MdotAlgDriver mdotDriver(
    helperObjs.realm, elementContinuityEqs);
  mdotDriver.register_open_mdot_corrector_alg(
    sierra::nalu::OPEN, surfPart, "mdot_open_correction");

  const double mdotInflow = -124.0;
  const double mdotOpen = 100.0;
  // 'surface_1' contains all the 6 faces of the hex element, therefore, 24
  // integration points in total where the mass flux is distributed.
  const double mdotCorrection = (mdotInflow + mdotOpen) / 24.0;
  // Correction algorithm is only applied to 'surface_5', therefore, only on 4
  // integration points over 1 face
  const double mdotOpenPost = -mdotCorrection * 4.0;
  mdotDriver.pre_work();
  mdotDriver.add_inflow_mdot(mdotInflow);
  mdotDriver.add_open_mdot(mdotOpen);
  mdotDriver.add_density_accumulation(0.0);
  mdotDriver.post_work();

  EXPECT_NEAR(mdotDriver.mdot_open_post(), mdotOpenPost, 1.0e-15);
  EXPECT_NEAR(mdotDriver.mdot_open_correction(), mdotCorrection, 1.0e-15);
}

TEST_F(MomentumEdgeHex8Mesh, NGP_mdot_inflow)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  stk::mesh::field_fill(1.0, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  stk::mesh::field_fill(1.0, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  auto* part = meta_->get_part("surface_5");
  auto* surfPart = part->subsets()[0];
  const bool elementContinuityEqs = true;
  const bool useShifted = true;

  sierra::nalu::MdotAlgDriver mdotDriver(
    helperObjs.realm, elementContinuityEqs);
  mdotDriver.register_face_algorithm<sierra::nalu::MdotInflowAlg>(
    sierra::nalu::INFLOW, surfPart, "mdot_inflow", mdotDriver, useShifted);
  mdotDriver.execute();

  EXPECT_NEAR(mdotDriver.mdot_inflow(), -1.0, 1.0e-15);
}

TEST_F(MomentumEdgeHex8Mesh, NGP_mdot_open_edge)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  fill_mesh_and_init_fields(doPerturb, generateSidesets);

  stk::mesh::field_fill(1.0, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  stk::mesh::field_fill(1.0, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  stk::mesh::field_fill(0.0, *pressure_);
  pressure_->modify_on_host();
  pressure_->sync_to_device();

  stk::mesh::field_fill(0.0, *dpdx_);
  dpdx_->modify_on_host();
  dpdx_->sync_to_device();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  auto* part = meta_->get_part("surface_6");
  auto* surfPart = part->subsets()[0];
  const bool elementContinuityEqs = true;
  const bool needCorrection = false;

  helperObjs.realm.solutionOptions_->activateOpenMdotCorrection_ = true;
  sierra::nalu::MdotAlgDriver mdotDriver(
    helperObjs.realm, elementContinuityEqs);
  mdotDriver.register_open_mdot_algorithm<sierra::nalu::MdotOpenEdgeAlg>(
    sierra::nalu::OPEN, surfPart, stk::topology::HEX_8, "mdot_open",
    needCorrection, mdotDriver);

  mdotDriver.execute();

  EXPECT_NEAR(mdotDriver.mdot_open(), 1.0, 1.0e-15);
}
