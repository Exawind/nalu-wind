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

#include "TurbViscKsgsAlgorithm.h"
#include "TurbKineticEnergyKsgsNodeSourceSuppAlg.h"
#include "TurbViscSmagorinskyAlgorithm.h"
#include "TurbViscWaleAlgorithm.h"

TEST_F(TestTurbulenceAlgorithm, turbviscksgsalgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  sierra::nalu::TurbViscKsgsAlgorithm alg(realm, meshPart_);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*tvisc_);
  const double gold_norm = 0.0285191520668428;
  EXPECT_NEAR(norm, gold_norm, tol);
}

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(TestTurbulenceAlgorithm, turbkineticenergyksgsnodesourcesuppalg)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Nodal execute
  auto& bulk = this->bulk();
  unit_test_algorithm_utils::TestSupplementalAlgorithmDriver assembleSuppAlgs(bulk);
  std::unique_ptr<sierra::nalu::SupplementalAlgorithm> suppalg(
    new sierra::nalu::TurbKineticEnergyKsgsNodeSourceSuppAlg(realm));
  assembleSuppAlgs.activeSuppAlgs_.push_back(suppalg.get());
  assembleSuppAlgs.nodal_execute();

  // Perform tests
  const double tol = 1e-12;
  const double lhs_norm = assembleSuppAlgs.get_lhs_norm();
  const double rhs_norm = assembleSuppAlgs.get_rhs_norm();
  const double lhs_gold_norm = 0.2469566986377040;
  const double rhs_gold_norm = 121.74000141151851;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}

#endif

TEST_F(TestTurbulenceAlgorithm, turbviscsmagorinskyalgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  sierra::nalu::TurbViscSmagorinskyAlgorithm alg(realm, meshPart_);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*tvisc_);
  const double gold_norm = 0.0015635636790984;
  EXPECT_NEAR(norm, gold_norm, tol);
}

TEST_F(TestTurbulenceAlgorithm, turbviscwalealgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  sierra::nalu::TurbViscWaleAlgorithm alg(realm, meshPart_);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*tvisc_);
  const double gold_norm = 0.0125168821438404;
  EXPECT_NEAR(norm, gold_norm, tol);
}
