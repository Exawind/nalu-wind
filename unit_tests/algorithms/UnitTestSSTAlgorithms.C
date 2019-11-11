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

#include "ComputeSSTMaxLengthScaleElemAlgorithm.h"

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
