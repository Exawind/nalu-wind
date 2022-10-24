// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "UnitTestAlgorithm.h"
#include "UnitTestKokkosUtils.h"
#include "UnitTestFieldUtils.h"
#include "UnitTestAlgorithmUtils.h"

#include "SolutionOptions.h"
#include "TurbKineticEnergyRodiNodeSourceSuppAlg.h"

#if !defined(KOKKOS_ENABLE_GPU)

TEST_F(TestTurbulenceAlgorithm, turbkineticenergyrodinodesourcesuppalg)
{
  sierra::nalu::Realm& realm = this->create_realm();

  const int nprocs = this->bulk().parallel_size();
  std::string meshSpec = "generated:1x1x" + std::to_string(nprocs);
  fill_mesh_and_init_fields(meshSpec);

  // set solution options
  realm.solutionOptions_->gravity_.resize(3);
  realm.solutionOptions_->gravity_[0] = 10.0;
  realm.solutionOptions_->gravity_[1] = -10.0;
  realm.solutionOptions_->gravity_[2] = 5.0;
  realm.solutionOptions_->turbPrMap_["enthalpy"] = 0.60;
  realm.solutionOptions_->thermalExpansionCoeff_ = 3.0e-3;

  // Nodal execute
  auto& bulk = this->bulk();
  unit_test_algorithm_utils::TestSupplementalAlgorithmDriver assembleSuppAlgs(
    bulk);
  std::unique_ptr<sierra::nalu::SupplementalAlgorithm> suppalg(
    new sierra::nalu::TurbKineticEnergyRodiNodeSourceSuppAlg(realm));
  assembleSuppAlgs.activeSuppAlgs_.push_back(suppalg.get());
  assembleSuppAlgs.nodal_execute();

  // Perform tests
  const double tol = 1e-14;
  const double lhs_norm = assembleSuppAlgs.get_lhs_norm();
  const double rhs_norm = assembleSuppAlgs.get_rhs_norm();
  const double lhs_gold_norm = 0.0;
  const double rhs_gold_norm =
    nprocs == 1 ? 0.00030133929246584567 : 0.0002760523778561557;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}

#endif
