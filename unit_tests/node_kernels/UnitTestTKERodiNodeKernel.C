// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernels/UnitTestKernelUtils.h"
#include "SolutionOptions.h"
#include "UnitTestUtils.h"
#include "UnitTestHelperObjects.h"

#include "node_kernels/TKERodiNodeKernel.h"

TEST_F(KsgsKernelHex8Mesh, NGP_turb_kenetic_energy_Rodi)
{
  // Only execute for 1 processor runs
  if (bulk_->parallel_size() > 1)
    return;

  const int nprocs = bulk_->parallel_size();

  std::mt19937 rng;
  rng.seed(0); // fixed seed

  fill_mesh_and_init_fields(false, false, true);

  unit_test_utils::NodeHelperObjects helperObjs(
    bulk_, stk::topology::HEX_8, 3, partVec_[0]);

  // set solution options
  sierra::nalu::SolutionOptions* solnOpts =
    helperObjs.nodeAlg->realm_.solutionOptions_;
  solnOpts->gravity_.resize(3);
  solnOpts->gravity_[0] = 10.0;
  solnOpts->gravity_[1] = -10.0;
  solnOpts->gravity_[2] = 5.0;
  solnOpts->turbPrMap_["enthalpy"] = 0.60;
  solnOpts->thermalExpansionCoeff_ = 3.0e-3;

  solnOpts_.gravity_.resize(3);
  solnOpts_.gravity_[0] = 10.0;
  solnOpts_.gravity_[1] = -10.0;
  solnOpts_.gravity_[2] = 5.0;
  solnOpts_.turbPrMap_["enthalpy"] = 0.60;
  solnOpts_.thermalExpansionCoeff_ = 3.0e-3;

  helperObjs.nodeAlg->add_kernel<sierra::nalu::TKERodiNodeKernel>(
    bulk_->mesh_meta_data(), *solnOpts);

  helperObjs.execute();

  EXPECT_EQ(helperObjs.linsys->lhs_.extent(0), 24u);
  EXPECT_EQ(helperObjs.linsys->lhs_.extent(1), 24u);
  EXPECT_EQ(helperObjs.linsys->rhs_.extent(0), 24u);

  Kokkos::deep_copy(helperObjs.linsys->hostlhs_, helperObjs.linsys->lhs_);
  Kokkos::deep_copy(helperObjs.linsys->hostrhs_, helperObjs.linsys->rhs_);

  auto lhs = helperObjs.linsys->hostlhs_;
  auto rhs = helperObjs.linsys->hostrhs_;

  double lhs_norm = 0, rhs_norm = 0;
  for (unsigned i = 0; i < rhs.extent(0); ++i)
    rhs_norm += rhs(i) * rhs(i);
  for (unsigned i = 0; i < lhs.extent(0); ++i)
    for (unsigned j = 0; j < lhs.extent(1); ++j)
      lhs_norm += lhs(i, j) * lhs(i, j);

  lhs_norm = std::sqrt(lhs_norm / (lhs.extent(0) * lhs.extent(1) / 3));
  rhs_norm = std::sqrt(rhs_norm / (rhs.extent(0) / 3));

  const double tol = 1e-14;
  const double lhs_gold_norm = 0.0;
  const double rhs_gold_norm =
    nprocs == 1 ? 0.00030133929246584567 : 0.0002760523778561557;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}
