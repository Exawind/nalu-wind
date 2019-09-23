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
