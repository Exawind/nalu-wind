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

#include "ComputeSSTMaxLengthScaleElemAlgorithm.h"
#include "TurbViscSSTAlgorithm.h"
#include "EffectiveSSTDiffFluxCoeffAlgorithm.h"
#include "TurbKineticEnergySSTNodeSourceSuppAlg.h"
#include "TurbKineticEnergySSTDESNodeSourceSuppAlg.h"
#include "SpecificDissipationRateSSTNodeSourceSuppAlg.h"

TEST_F(TestTurbulenceAlgorithm, computesstmaxlengthscaleelemalgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  sierra::nalu::ComputeSSTMaxLengthScaleElemAlgorithm alg(realm, meshPart_);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*maxLengthScale_);
  const double gold_norm = 1.0;
  EXPECT_NEAR(norm, gold_norm, tol);
}

TEST_F(TestTurbulenceAlgorithm, testturbviscsstalgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  sierra::nalu::TurbViscSSTAlgorithm alg(realm, meshPart_);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*tvisc_);
  const double gold_norm = 0.41450173743648816;
  EXPECT_NEAR(norm, gold_norm, tol);
}

TEST_F(TestTurbulenceAlgorithm, effectivesstdifffluxcoeffalgorithm)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Execute
  const double sigmaOne = 0.85;
  const double sigmaTwo = 1.0;
  sierra::nalu::EffectiveSSTDiffFluxCoeffAlgorithm alg(realm, meshPart_, viscosity_, tvisc_, evisc_, sigmaOne, sigmaTwo);
  alg.execute();

  // Perform tests
  const double tol = 1e-14;
  double norm = field_norm(*evisc_);
  const double gold_norm = 2.2388729777522056;
  EXPECT_NEAR(norm, gold_norm, tol);
}

#ifndef KOKKOS_ENABLE_CUDA

TEST_F(TestTurbulenceAlgorithm, turbkineticenergysstnodesourcesuppalg)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Nodal execute
  auto& bulk = this->bulk();
  unit_test_algorithm_utils::TestSupplementalAlgorithmDriver assembleSuppAlgs(bulk);
  std::unique_ptr<sierra::nalu::SupplementalAlgorithm> suppalg(
    new sierra::nalu::TurbKineticEnergySSTNodeSourceSuppAlg(realm));
  assembleSuppAlgs.activeSuppAlgs_.push_back(suppalg.get());
  assembleSuppAlgs.nodal_execute();

  // Perform tests
  const double tol = 1e-14;
  const double lhs_norm = assembleSuppAlgs.get_lhs_norm();
  const double rhs_norm = assembleSuppAlgs.get_rhs_norm();
  const double lhs_gold_norm = 0.014564571309018231;
  const double rhs_gold_norm = 10.448455484893881;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}

TEST_F(TestTurbulenceAlgorithm, turbkineticenergysstdesnodesourcesuppalg)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Nodal execute
  auto& bulk = this->bulk();
  unit_test_algorithm_utils::TestSupplementalAlgorithmDriver assembleSuppAlgs(bulk);
  std::unique_ptr<sierra::nalu::SupplementalAlgorithm> suppalg(
    new sierra::nalu::TurbKineticEnergySSTDESNodeSourceSuppAlg(realm));
  assembleSuppAlgs.activeSuppAlgs_.push_back(suppalg.get());
  assembleSuppAlgs.nodal_execute();

  // Perform tests
  const double tol = 1e-12;
  const double lhs_norm = assembleSuppAlgs.get_lhs_norm();
  const double rhs_norm = assembleSuppAlgs.get_rhs_norm();
  const double lhs_gold_norm = 0.56267322649917717;
  const double rhs_gold_norm = 277.31214722069728;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}

TEST_F(TestTurbulenceAlgorithm, specificdissipationratesstnodesourcesuppalg)
{
  sierra::nalu::Realm& realm = this->create_realm();

  fill_mesh_and_init_fields();

  // Nodal execute
  auto& bulk = this->bulk();
  unit_test_algorithm_utils::TestSupplementalAlgorithmDriver assembleSuppAlgs(bulk);
  std::unique_ptr<sierra::nalu::SupplementalAlgorithm> suppalg(
    new sierra::nalu::SpecificDissipationRateSSTNodeSourceSuppAlg(realm));
  assembleSuppAlgs.activeSuppAlgs_.push_back(suppalg.get());
  assembleSuppAlgs.nodal_execute();

  // Perform tests
  const double tol = 1e-14;
  const double lhs_norm = assembleSuppAlgs.get_lhs_norm();
  const double rhs_norm = assembleSuppAlgs.get_rhs_norm();
  const double lhs_gold_norm = 0.027222983025572127;
  const double rhs_gold_norm = 2.7483377905404858;
  EXPECT_NEAR(lhs_norm, lhs_gold_norm, tol);
  EXPECT_NEAR(rhs_norm, rhs_gold_norm, tol);
}

#endif

