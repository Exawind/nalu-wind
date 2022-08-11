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

#include "ngp_algorithms/TurbViscKOAlg.h"

TEST_F(KOKernelHex8Mesh, NGP_turb_visc_ko_alg)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  KOKernelHex8Mesh::fill_mesh_and_init_fields();

  // Initialize turbulence parameters in solution options
  solnOpts_.initialize_turbulence_constants();

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);

  sierra::nalu::TurbViscKOAlg TurbViscKOAlg(
    helperObjs.realm, partVec_[0], tvisc_);

  TurbViscKOAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpTvisc = fieldMgr.get_field<double>(tvisc_->mesh_meta_data_ordinal());
  ngpTvisc.modify_on_device();
  ngpTvisc.sync_to_host();

  {
    std::vector<double> expectedValues = {
      0.46763636363636363,  0.20271993944117145,  0.34820558301166943,
      0.11992191196521775,  0.20271993944117145,  0.083672109203660194,
      0.074293947075371264, 0.031038745209797804,
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
