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

#include "ngp_algorithms/TurbViscSSTAlg.h"
#include "ngp_algorithms/TurbViscSSTLRAlg.h"

TEST_F(SSTKernelHex8Mesh, NGP_turb_visc_sst_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  SSTKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TurbViscSSTAlg TurbViscSSTAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscSSTAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {1,
                                          0.58778525229247314,
                                          0.82554938136626144,
                                          0.33398788772269555,
                                          0.58778525229247314,
                                          0.34549150281252627,
                                          0.32219673776462515,
                                          0.19411612710296283};

    const double tol = 1.0e-15;

    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* tvisc = stk::mesh::field_data(*tvisc_, node);
        EXPECT_NEAR(tvisc[0], expectedValues[ii++], tol);
      }
  }
}

TEST_F(SSTKernelHex8Mesh, NGP_turb_visc_sstlr_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  SSTKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TurbViscSSTAlg TurbViscSSTLRAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscSSTLRAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {1,
                                          0.58778525229247314,
                                          0.82554938136626144,
                                          0.33398788772269555,
                                          0.58778525229247314,
                                          0.34549150281252627,
                                          0.32219673776462515,
                                          0.19411612710296283};

    const double tol = 1.0e-15;

    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* tvisc = stk::mesh::field_data(*tvisc_, node);
        EXPECT_NEAR(tvisc[0], expectedValues[ii++], tol);
      }
  }
}

TEST_F(AMSKernelHex8Mesh, NGP_turb_visc_sstams_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  AMSKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TurbViscSSTAlg TurbViscSSTAlg(
    helperObjs.realm, partVec_[0], tvisc_, true);

  TurbViscSSTAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      1, 1, 1.4045084971874737,  0.62203263379915574,
      1, 1, 0.93257499863740057, 0.57277822202446083,
    };

    const double tol = 1.0e-15;

    stk::mesh::Selector sel = meta_->universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    int ii = 0;
    for (const auto* b : bkts)
      for (const auto node : *b) {
        const double* tvisc = stk::mesh::field_data(*tvisc_, node);
        EXPECT_NEAR(tvisc[0], expectedValues[ii++], tol);
      }
  }
}
