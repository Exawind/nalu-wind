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

#include "ngp_algorithms/TurbViscKEAlg.h"

TEST_F(KEKernelHex8Mesh, NGP_turb_visc_ke_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  KEKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TurbViscKEAlg TurbViscKEAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscKEAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      0.0040927529208101276, 0.004756620887334956,  0.0047455106720311118,
      0.0042834431659046351, 0.0024056598081291371, 0.0027958716083218245,
      0.0016322044534763486, 0.0017904108086812805,
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
