/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "UnitTestAlgorithm.h"
#include "UnitTestKokkosUtils.h"
#include "UnitTestFieldUtils.h"
#include "UnitTestAlgorithmUtils.h"

#include "SolutionOptions.h"
#include "TurbKineticEnergyRodiNodeSourceSuppAlg.h"

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(TestTurbulenceAlgorithm, turbkineticenergyrodinodesourcesuppalg)
{
  sierra::nalu::Realm& realm = this->create_realm();
  
  fill_mesh_and_init_fields();

  // set solution options
  realm.solutionOptions_->gravity_.resize(3);
  realm.solutionOptions_->gravity_[0] = 10.0;
  realm.solutionOptions_->gravity_[1] = -10.0;
  realm.solutionOptions_->gravity_[2] = 5.0;
  realm.solutionOptions_->turbPrMap_["enthalpy"] = 0.60;
  realm.solutionOptions_->thermalExpansionCoeff_ = 3.0e-3;
  
  // Nodal execute
  auto& bulk = this->bulk();
  unit_test_algorithm_utils::TestSupplementalAlgorithmDriver assembleSuppAlgs(bulk);
  std::unique_ptr<sierra::nalu::SupplementalAlgorithm> suppalg(
    new sierra::nalu::TurbKineticEnergyRodiNodeSourceSuppAlg(realm));
  assembleSuppAlgs.activeSuppAlgs_.push_back(suppalg.get());
  assembleSuppAlgs.nodal_execute();

  // Perform tests
  const double tol = 1e-14;
  const double lhs_norm = assembleSuppAlgs.get_lhs_norm();
  const double rhs_norm = assembleSuppAlgs.get_rhs_norm();
  const double lhs_gold_norm = 0.0;
  const double rhs_gold_norm = 0.000244820116732111;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}

#endif

