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

#include "ngp_algorithms/TurbViscKsgsAlg.h"

TEST_F(KsgsKernelHex8Mesh, NGP_turb_visc_ksgs_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  KsgsKernelHex8Mesh::fill_mesh_and_init_fields(false, false, false);

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TurbViscKsgsAlg TurbViscKsgsAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscKsgsAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      0.060528340469568467, 0.035577665873750018, 0.042163789612795696,
      0.023265644281201679, 0.035577665873750018, 0.020912027311579463,
      0.023265644281201679, 0.013122616366371343};

    const double tol = 1.0e-16;
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
