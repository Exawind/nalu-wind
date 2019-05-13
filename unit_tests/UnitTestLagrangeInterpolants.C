#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <element_promotion/LagrangeBasis.h>

#include "UnitTestUtils.h"

namespace sierra { namespace nalu {

TEST(Lagrange1D, kronecker)
{
  std::vector<double> nodeLocs = { -150.0, -.2, 0.3333, 0.5, -4, 2.0};
  auto basis = Lagrange1D(nodeLocs.data(), nodeLocs.size() - 1);

  for (unsigned k = 0; k < nodeLocs.size(); ++k) {
    for (unsigned j = 0; j < nodeLocs.size(); ++j) {
      EXPECT_DOUBLE_EQ(basis.interpolation_weight(nodeLocs[k],j), (k == j) ? 1.0 : 0);
    }
  }
}

TEST(Lagrange1D, linear_interpolants)
{
  std::vector<double> nodeLocs = {-1.0, +1.0};
  auto basis = Lagrange1D(nodeLocs.data(), nodeLocs.size() - 1);

  const double x = 0.243;
  EXPECT_DOUBLE_EQ(basis.interpolation_weight(x, 0), 0.5 * (1 - x));
  EXPECT_DOUBLE_EQ(basis.interpolation_weight(x, 1), 0.5 * (1 + x));
}

TEST(Lagrange1D, linear_derivatives)
{
  std::vector<double> nodeLocs = {-1.0, +1.0};
  auto basis = Lagrange1D(nodeLocs.data(), nodeLocs.size() - 1);

  const double x = 0.57462;
  EXPECT_DOUBLE_EQ(basis.derivative_weight(x, 0), -0.5);
  EXPECT_DOUBLE_EQ(basis.derivative_weight(x, 1), +0.5);
}

TEST(Lagrange1D, quadratic_interpolants)
{
  std::vector<double> nodeLocs = {-1.0, 0.0, +1.0};
  auto basis = Lagrange1D(nodeLocs.data(), nodeLocs.size() - 1);

  const double x = 0.24334534;
  EXPECT_DOUBLE_EQ(basis.interpolation_weight(x, 0), -0.5*x*(1.0-x));
  EXPECT_DOUBLE_EQ(basis.interpolation_weight(x, 1), (1 - x) * (1 + x));
  EXPECT_DOUBLE_EQ(basis.interpolation_weight(x, 2), +0.5*x*(1.0+x));
}

TEST(Lagrange1D, quadratic_derivatives)
{
  std::vector<double> nodeLocs = {-1.0, 0.0, +1.0};
  auto basis = Lagrange1D(nodeLocs.data(), nodeLocs.size() - 1);

  const double x = -0.907097;
  EXPECT_DOUBLE_EQ(basis.derivative_weight(x, 0), -0.5 + x);
  EXPECT_DOUBLE_EQ(basis.derivative_weight(x, 1), -2 * x);
  EXPECT_DOUBLE_EQ(basis.derivative_weight(x, 2), +0.5 + x);
}
}}
