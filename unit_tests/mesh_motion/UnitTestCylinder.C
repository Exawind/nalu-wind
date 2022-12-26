#include <gtest/gtest.h>
#include <limits>
#include "Realm.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "UnitTestRealm.h"
#include "UnitTestUtils.h"

namespace {
const double testTol = 1e-12;

TEST(meshMotion, CYLINDER)
{
  // create realm
  unit_test_utils::NaluTest naluObj;
  sierra::nalu::Realm& realm = naluObj.create_realm();
  // realm.solutionOptions_->meshTransformation_ = true;

  sierra::nalu::TimeIntegrator timeIntegrator;
  timeIntegrator.secondOrderTimeAccurate_ = false;
  realm.timeIntegrator_ = &timeIntegrator;

  // register mesh motion fields and initialize coordinate fields
  realm.register_nodal_fields(&(realm.meta_data().universal_part()));

  // create mesh and get dimensions
  const int imax = 5;
  const int jmax = 10;
  const int kmax = 15;
  const double innerRad = 1.0;
  const double outerRad = 2.0;
  unit_test_utils::fill_hex8_cylinder_mesh(
    innerRad, outerRad, imax, jmax, kmax, realm.bulk_data());

  unit_test_utils::dump_mesh(realm.bulk_data(), {}, "cylinder2.e");

  EXPECT_NEAR(1.0e-12, 0.0, testTol);
}
} // namespace