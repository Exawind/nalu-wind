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

#include "TurbViscSmagorinskyAlgorithm.h"
#include "TurbViscWaleAlgorithm.h"

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
  const double gold_norm = 0.0094154596233012953;
  EXPECT_NEAR(norm, gold_norm, tol);
}
