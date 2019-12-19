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

#include "ngp_algorithms/EffSSTDiffFluxCoeffAlg.h"

TEST_F(KsgsKernelHex8Mesh, NGP_eff_sst_diff_flux_coeff_alg)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  LowMachKernelHex8Mesh::fill_mesh_and_init_fields();

  const double sigmaOne = 0.85;
  const double sigmaTwo = 1.0;

  stk::mesh::field_fill(sigmaOne, *viscosity_);
  stk::mesh::field_fill(sigmaTwo, *tvisc_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  sierra::nalu::EffSSTDiffFluxCoeffAlg diffFluxAlg(
    helperObjs.realm, partVec_[0], viscosity_, tvisc_, evisc_,
    sigmaOne, sigmaTwo);
  diffFluxAlg.execute();

  const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
  auto ngpEvisc = fieldMgr.get_field<double>(evisc_->mesh_meta_data_ordinal());
  ngpEvisc.modify_on_device();
  ngpEvisc.sync_to_host();

  {
    const double tol = 1.0e-16;
    stk::mesh::Selector sel = meta_.universal_part();
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* evisc = stk::mesh::field_data(*evisc_, node);
        EXPECT_NEAR(evisc[0], 1.85, tol);
      }
  }
}
