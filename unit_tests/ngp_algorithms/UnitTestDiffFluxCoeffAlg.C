/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "ngp_algorithms/EffDiffFluxCoeffAlg.h"

TEST_F(KsgsKernelHex8Mesh, NGP_eff_diff_flux_coeff)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  LowMachKernelHex8Mesh::fill_mesh_and_init_fields();

  const double sigmaLam = 0.3;
  const double sigmaTurb = 0.7;
  const bool isTurbulent = true;

  stk::mesh::field_fill(sigmaLam, *viscosity_);
  stk::mesh::field_fill(sigmaTurb, *tvisc_);

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  sierra::nalu::EffDiffFluxCoeffAlg diffFluxAlg(
    helperObjs.realm, partVec_[0], viscosity_, tvisc_, evisc_,
    sigmaLam, sigmaTurb, isTurbulent);
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
        EXPECT_NEAR(evisc[0], 2.0, tol);
      }
  }
}
