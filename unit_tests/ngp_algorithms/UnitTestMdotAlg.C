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

#include "ngp_algorithms/MdotEdgeAlg.h"

TEST_F(MomentumEdgeHex8Mesh, NGP_mdot_calc_edge)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // Setup solution options for default advection kernel
  solnOpts_.meshMotion_ = false;
  solnOpts_.meshDeformation_ = false;
  solnOpts_.externalMeshDeformation_ = false;
  solnOpts_.mdotInterpRhoUTogether_ = true;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.interiorPartVec_.push_back(partVec_[0]);

  stk::mesh::field_fill(1.0, *density_);
  stk::mesh::field_fill(10.0, *velocity_);
  stk::mesh::field_fill(0.0, *pressure_);
  stk::mesh::field_fill(0.0, *dpdx_);
  stk::mesh::field_fill(0.0, *massFlowRateEdge_);

  sierra::nalu::MdotEdgeAlg mdotAlg(helperObjs.realm, partVec_[0]);
  mdotAlg.execute();

  const auto& fieldMgr = helperObjs.realm.ngp_field_manager();
  auto ngpMdot =
    fieldMgr.get_field<double>(massFlowRateEdge_->mesh_meta_data_ordinal());
  ngpMdot.modify_on_device();
  ngpMdot.sync_to_host();

  {
    const double tol = 1.0e-14;
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::EDGE_RANK, sel);

    for (const auto* b: bkts)
      for (const auto edge: *b) {
        const double* mdot = stk::mesh::field_data(*massFlowRateEdge_, edge);
        EXPECT_NEAR(mdot[0], 2.5, tol);
      }
  }
}
