/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "ngp_algorithms/UnitTestNgpAlgUtils.h"

#include "ngp_algorithms/TurbKineticEnergyWallAlg.h"
#include "ngp_algorithms/NgpAlgDriver.h"
#include "kernels/UnitTestKernelUtils.h"

#include "ComputeTurbKineticEnergyWallFunctionAlgorithm.h"

TEST_F(KsgsKernelHex8Mesh, NGP_turb_kin_enrg_wall_alg)
{
  // Only execute for 1 processor runs
  if (bulk_.parallel_size() > 1) return;

  const bool doPerturb = false;
  const bool generateSidesets = true;
  const bool perturb_turbulent_viscosity_and_dual_nodal_volume = false;
  KsgsKernelHex8Mesh::fill_mesh_and_init_fields(doPerturb, generateSidesets,
    perturb_turbulent_viscosity_and_dual_nodal_volume);

  const double xCoeff = 2.0;
  const double yCoeff = 2.0;
  const double zCoeff = 2.0;

  auto* part = meta_.get_part("surface_5");
  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::QUAD_4, 1, part);
  unit_test_alg_utils::linear_scalar_field(bulk_, *coordinates_, *tke_,
                                           xCoeff, yCoeff, zCoeff);
  stk::mesh::field_fill(0.0, *tke_);

  stk::mesh::Part* surfPart = part->subsets()[0];
  sierra::nalu::NgpAlgDriver algDriver(helperObjs.realm);

  algDriver.register_face_algorithm<sierra::nalu::TurbKineticEnergyWallAlg>(
    sierra::nalu::WALL, surfPart, "turb_kinetic_energy_wall");
  algDriver.execute();

  { 
    const auto& fieldMgr = helperObjs.realm.mesh_info().ngp_field_manager();
    auto ngpbcassem = fieldMgr.get_field<double>(bcAssemTke_->mesh_meta_data_ordinal());
    ngpbcassem.modify_on_device();
    ngpbcassem.sync_to_host();

    const double expectedValue = 0.00093851756819561495;
    const double tol = 1.0e-16;
    stk::mesh::Selector sel(*part);
    const auto& bkts = bulk_.get_buckets(stk::topology::NODE_RANK, sel);

    for (const auto* b: bkts)
      for (const auto node: *b) {
        const double* bc_tke = stk::mesh::field_data(*bcAssemTke_, node);
        EXPECT_NEAR(bc_tke[0], expectedValue, tol);
      }
  }
}

