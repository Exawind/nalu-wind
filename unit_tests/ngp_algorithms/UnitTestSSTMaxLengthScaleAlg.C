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
#include "ngp_algorithms/SSTMaxLengthScaleAlg.h"
#include "ngp_algorithms/SSTMaxLengthScaleDriver.h"
#include "utils/StkHelpers.h"

TEST_F(SSTKernelHex8Mesh, NGP_SST_Max_Length_Scale)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  fill_mesh_and_init_fields();

  // zero out fields
  stk::mesh::field_fill(0.0, *maxLengthScale_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  // Force computation of edge area vector
  helperObjs.realm.realmUsesEdges_ = true;

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto& ngpMaxLen = fieldMgr.get_field<double>(
      maxLengthScale_->mesh_meta_data_ordinal());

  sierra::nalu::SSTMaxLengthScaleDriver AlgDriver(helperObjs.realm);

  AlgDriver.register_elem_algorithm<sierra::nalu::SSTMaxLengthScaleAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "SSTMaxLen");

  AlgDriver.execute();

  ngpMaxLen.modify_on_device();
  ngpMaxLen.sync_to_host();

  const double tol = 1.0e-16;
  stk::mesh::Selector sel = meta_.universal_part();

  {
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);
    int counter = 0;
    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* mLen = stk::mesh::field_data(*maxLengthScale_, node);
        EXPECT_NEAR(1.0, mLen[0], tol);
        counter++;
      }
    EXPECT_EQ(counter, 8);
  }
}
