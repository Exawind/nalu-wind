// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <random>

#include "kernels/UnitTestKernelUtils.h"
#include "UnitTestHelperObjects.h"

#include "AlgTraits.h"
#include "ngp_algorithms/CourantReAlg.h"
#include "ngp_algorithms/CourantReAlgDriver.h"
#include "utils/StkHelpers.h"

TEST_F(MomentumKernelHex8Mesh, NGP_courant_reynolds)
{
  auto& elemCourant = meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "element_courant");
  auto& elemReynolds = meta_->declare_field<GenericFieldType>(
    stk::topology::ELEM_RANK, "element_reynolds");
  stk::mesh::put_field_on_mesh(
    elemCourant, meta_->universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(
    elemReynolds, meta_->universal_part(), 1, nullptr);
  fill_mesh_and_init_fields();

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> rand_num(-1.0, 1.0);

  const double dt = 0.1;
  const double velVal = 10.0 + rand_num(rng);
  const double rhoVal = 1.0 + 0.1 * rand_num(rng);
  const double viscVal = 1.0e-5 * (1.0 + rand_num(rng));

  const double reyNum = velVal / (viscVal / rhoVal + 1.0e-16);
  const double cfl = velVal * dt;

  stk::mesh::field_fill(velVal, *velocity_);
  velocity_->modify_on_host();
  velocity_->sync_to_device();

  stk::mesh::field_fill(rhoVal, *density_);
  density_->modify_on_host();
  density_->sync_to_device();

  stk::mesh::field_fill(viscVal, *viscosity_);
  viscosity_->modify_on_host();
  viscosity_->sync_to_device();

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.timeStepN_ = dt;
  timeIntegrator.timeStepNm1_ = dt;
  timeIntegrator.gamma1_ = 1.0;
  timeIntegrator.gamma2_ = -1.0;
  timeIntegrator.gamma3_ = 0.0;

  unit_test_utils::HelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 1, partVec_[0]);
  helperObjs.realm.timeIntegrator_ = &timeIntegrator;

  sierra::nalu::CourantReAlgDriver algDriver(helperObjs.realm);
  algDriver.register_elem_algorithm<sierra::nalu::CourantReAlg>(
    sierra::nalu::INTERIOR, partVec_[0], "courant_reynolds", algDriver);

  algDriver.execute();

  EXPECT_NEAR(helperObjs.realm.maxCourant_, cfl, 1.0e-14);
  EXPECT_NEAR(helperObjs.realm.maxReynolds_, reyNum, 1.0e-14);
}
