#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <element_promotion/QuadratureRule.h>

#include "UnitTestUtils.h"

TEST(QuadratureRule, lobatto_rule)
{
  std::vector<double> abscissae;
  std::vector<double> weights;
  std::vector<double> exactX;
  std::vector<double> exactW;

  std::tie(abscissae,weights) = sierra::nalu::gauss_lobatto_legendre_rule(3);
  exactX = {-1.0, 0.0, +1.0};
  exactW = { 1.0/3.0, 4.0/3.0, 1.0/3.0 };

  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }

  std::tie(abscissae,weights) = sierra::nalu::gauss_lobatto_legendre_rule(4);
  double xl0 = std::sqrt(5.0)/5.0;
  double xw0 = 5.0/6.0;
  double xw1 = 1.0/6.0;
  exactX = {-1.0, -xl0, +xl0, +1.0};
  exactW = { xw1, xw0, xw0, xw1 }; // sums to 2

  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }

  std::tie(abscissae,weights) = sierra::nalu::gauss_lobatto_legendre_rule(5);
  xl0 = std::sqrt(21.0)/7.0;
  xw0 = 32.0/45.0;
  xw1 = 49.0/90.0;
  double xw2 = 1.0/10.0;
  exactX = {-1.0, -xl0, 0.0, xl0, +1.0};
  exactW = { xw2, xw1, xw0, xw1, xw2 }; // sums to 2

  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }

  std::tie(abscissae,weights) = sierra::nalu::gauss_lobatto_legendre_rule(6);
  xl0 = std::sqrt((7.0-2.0*std::sqrt(7.0))/21.0);
  double xl1 = std::sqrt((7.0+2.0*std::sqrt(7.0))/21.0);
  xw0 = (14.0+std::sqrt(7.0))/30.0;
  xw1 = (14.0-std::sqrt(7.0))/30.0;
  xw2 = 1.0/15.0;
  exactX = {-1.0, -xl1, -xl0, xl0, +xl1, +1.0};
  exactW = { xw2, xw1, xw0, xw0, xw1, xw2 }; // sums to 2


  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }
}

TEST(QuadratureRule, gauss_legendre_rule)
{
  std::vector<double> abscissae;
  std::vector<double> weights;
  std::vector<double> exactX;
  std::vector<double> exactW;

  std::tie(abscissae,weights) = sierra::nalu::gauss_legendre_rule(2);
  exactX = {-std::sqrt(3.0)/3.0, std::sqrt(3.0)/3.0 };
  exactW = { 1.0, 1.0 };

  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }

  std::tie(abscissae,weights) = sierra::nalu::gauss_legendre_rule(3);
  exactX = { -std::sqrt(3.0/5.0), 0.0, std::sqrt(3.0/5.0) };
  exactW = { 5.0/9.0, 8.0/9.0,  5.0/9.0 };

  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }

  std::tie(abscissae,weights) = sierra::nalu::gauss_legendre_rule(4);
  exactX = {
      -std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0)),
      -std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
      +std::sqrt(3.0/7.0-2.0/7.0*std::sqrt(6.0/5.0)),
      +std::sqrt(3.0/7.0+2.0/7.0*std::sqrt(6.0/5.0))
  };

  exactW = {
      (18.0-std::sqrt(30.0))/36.0,
      (18.0+std::sqrt(30.0))/36.0,
      (18.0+std::sqrt(30.0))/36.0,
      (18.0-std::sqrt(30.0))/36.0
  };


  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }

  std::tie(abscissae,weights) = sierra::nalu::gauss_legendre_rule(5);
  exactX = {
      -std::sqrt(245.0+14.0*std::sqrt(70.0))/21.0,
      -std::sqrt(245.0-14.0*std::sqrt(70.0))/21.0,
      0.0,
      +std::sqrt(245.0-14.0*std::sqrt(70.0))/21.0,
      +std::sqrt(245.0+14.0*std::sqrt(70.0))/21.0
  };

  exactW = {
      (322.0-13.0*std::sqrt(70.0))/900.0,
      (322.0+13.0*std::sqrt(70.0))/900.0,
      128.0/225.0,
      (322.0+13.0*std::sqrt(70.0))/900.0,
      (322.0-13.0*std::sqrt(70.0))/900.0
  };


  for (unsigned j = 0; j < abscissae.size(); ++j) {
    EXPECT_NEAR(abscissae[j], exactX[j], tol);
    EXPECT_NEAR(weights[j], exactW[j], tol);
  }
}
