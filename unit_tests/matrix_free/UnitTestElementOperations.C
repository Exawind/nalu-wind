// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <cmath>
#include <random>

#include "matrix_free/Coefficients.h"
#include "matrix_free/ElementFluxIntegral.h"
#include "matrix_free/ElementGradient.h"
#include "matrix_free/ElementSCSInterpolate.h"
#include "matrix_free/ElementVolumeIntegral.h"
#include "matrix_free/GaussLegendreQuadratureRule.h"
#include "matrix_free/HexVertexCoordinates.h"
#include "matrix_free/LobattoQuadratureRule.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/KokkosViewTypes.h"
#include "ArrayND.h"
#include "StkSimdComparisons.h"
#include "gtest/gtest.h"

namespace sierra {
namespace nalu {
namespace matrix_free {

namespace {

double my_tol = 1.0e-10;

double
poly_val(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j] * std::pow(x, j);
  }
  return val;
}

double
poly_der(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 1; j < coeffs.size(); ++j) {
    val += j * coeffs[j] * std::pow(x, j - 1);
  }
  return val;
}

double
poly_int(std::vector<double> coeffs, double xlower, double xupper)
{
  double upper = 0.0;
  double lower = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    upper += coeffs[j] * std::pow(xupper, j + 1) / (j + 1.0);
    lower += coeffs[j] * std::pow(xlower, j + 1) / (j + 1.0);
  }
  return (upper - lower);
}

struct TensorPoly
{
  TensorPoly(int p)
  {
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    for (int j = 0; j < p + 1; ++j) {
      coeffsX.push_back(coeff(rng));
      coeffsY.push_back(coeff(rng));
      coeffsZ.push_back(coeff(rng));
    }
  }

  double operator()(double x, double y, double z)
  {
    return poly_val(coeffsX, x) * poly_val(coeffsY, y) * poly_val(coeffsZ, z);
  }

  double
  xyz_integral(double xl, double xr, double yl, double yr, double zl, double zr)
  {
    return poly_int(coeffsX, xl, xr) * poly_int(coeffsY, yl, yr) *
           poly_int(coeffsZ, zl, zr);
  }

  double der(int d, double x, double y, double z)
  {
    switch (d) {
    case 0:
      return poly_der(coeffsX, x) * poly_val(coeffsY, y) * poly_val(coeffsZ, z);
    case 1:
      return poly_val(coeffsX, x) * poly_der(coeffsY, y) * poly_val(coeffsZ, z);
    default:
      return poly_val(coeffsX, x) * poly_val(coeffsY, y) * poly_der(coeffsZ, z);
    }
  }

  std::vector<double> coeffsX;
  std::vector<double> coeffsY;
  std::vector<double> coeffsZ;
};

} // namespace

TEST(element_operations, scs_interp)
{
  constexpr int p = inst::P2;
  ArrayND<ftype[p + 1][p + 1][p + 1]> nodal_values;

  auto poly = TensorPoly(p);
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        nodal_values(k, j, i) =
          poly(GLL<p>::nodes[i], GLL<p>::nodes[j], GLL<p>::nodes[k]);
      }
    }
  }

  ArrayND<ftype[3][p + 1][p + 1][p + 1]> interp_values;
  interp_scs<p>(nodal_values, Coeffs<p>::Nt, interp_values);

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p; ++i) {
        ASSERT_DOUBLETYPE_NEAR(
          interp_values(0, i, k, j),
          poly(LGL<p>::nodes[i], GLL<p>::nodes[j], GLL<p>::nodes[k]), my_tol);
      }
    }
  }

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ASSERT_DOUBLETYPE_NEAR(
          interp_values(1, j, k, i),
          poly(GLL<p>::nodes[i], LGL<p>::nodes[j], GLL<p>::nodes[k]), my_tol);
      }
    }
  }

  for (int k = 0; k < p; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        ASSERT_DOUBLETYPE_NEAR(
          interp_values(2, k, j, i),
          poly(GLL<p>::nodes[i], GLL<p>::nodes[j], LGL<p>::nodes[k]), my_tol);
      }
    }
  }
}

TEST(element_operations, integrate_volume)
{
  constexpr int p = inst::P2;
  ArrayND<ftype[p + 1][p + 1][p + 1]> nodal_values;

  auto poly = TensorPoly(p);
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        nodal_values(k, j, i) =
          poly(GLL<p>::nodes[i], GLL<p>::nodes[j], GLL<p>::nodes[k]);
      }
    }
  }

  Kokkos::Array<double, p + 2> scs_end_loc;
  scs_end_loc[0] = -1;
  for (int j = 0; j < p; ++j) {
    scs_end_loc[j + 1] = LGL<p>::nodes[j];
  }
  scs_end_loc[p + 1] = +1;

  ArrayND<ftype[p + 1][p + 1][p + 1]> volumes;
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        volumes(k, j, i) = 0;
      }
    }
  }
  ArrayND<ftype[p + 1][p + 1][p + 1]> scratch;
  volume<p>(nodal_values, scratch, volumes);

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        const auto value = poly.xyz_integral(
          scs_end_loc[i], scs_end_loc[i + 1], scs_end_loc[j],
          scs_end_loc[j + 1], scs_end_loc[k], scs_end_loc[k + 1]);
        ASSERT_DOUBLETYPE_NEAR(volumes(k, j, i), value, my_tol);
      }
    }
  }
}

TEST(element_operations, nodal_grad)
{
  constexpr int p = inst::P2;
  ArrayND<ftype[p + 1][p + 1][p + 1]> nodal_values;

  ArrayND<ftype[p + 1][p + 1][p + 1][3]> xc;

  auto poly = TensorPoly(p);
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        const auto x = GLL<p>::nodes[i];
        const auto y = GLL<p>::nodes[j];
        const auto z = GLL<p>::nodes[k];
        xc(k, j, i, 0) = x;
        xc(k, j, i, 1) = y;
        xc(k, j, i, 2) = z;
        nodal_values(k, j, i) = poly(x, y, z);
      }
    }
  }

  auto box = hex_vertex_coordinates<p>(xc);
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        auto grad_approx = gradient_nodal<p>(box, nodal_values, k, j, i);

        const auto x = GLL<p>::nodes[i];
        const auto y = GLL<p>::nodes[j];
        const auto z = GLL<p>::nodes[k];

        ASSERT_DOUBLETYPE_NEAR(grad_approx[0], poly.der(0, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[1], poly.der(1, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[2], poly.der(2, x, y, z), my_tol);
      }
    }
  }
}

TEST(element_operations, fp_grad)
{
  constexpr int p = inst::P3;
  ArrayND<ftype[p + 1][p + 1][p + 1]> nodal_values;

  ArrayND<ftype[p + 1][p + 1][p + 1][3]> xc;

  auto poly = TensorPoly(p);
  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        const auto x = GLL<p>::nodes[i];
        const auto y = GLL<p>::nodes[j];
        const auto z = GLL<p>::nodes[k];
        xc(k, j, i, 0) = x;
        xc(k, j, i, 1) = y;
        xc(k, j, i, 2) = z;
        nodal_values(k, j, i) = poly(x, y, z);
      }
    }
  }

  auto box = hex_vertex_coordinates<p>(xc);
  for (int l = 0; l < p; ++l) {
    ArrayND<ftype[p + 1][p + 1]> nhat;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        nhat(s, r) = interp_scs<p, 0>(nodal_values, l, s, r);
      }
    }
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        auto grad_approx = gradient_scs<p, 0>(box, nodal_values, nhat, l, s, r);
        const auto x = LGL<p>::nodes[l];
        const auto y = GLL<p>::nodes[r];
        const auto z = GLL<p>::nodes[s];
        ASSERT_DOUBLETYPE_NEAR(grad_approx[0], poly.der(0, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[1], poly.der(1, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[2], poly.der(2, x, y, z), my_tol);
      }
    }
  }

  for (int l = 0; l < p; ++l) {
    ArrayND<ftype[p + 1][p + 1]> nhat;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        nhat(s, r) = interp_scs<p, 1>(nodal_values, l, s, r);
      }
    }
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        auto grad_approx = gradient_scs<p, 1>(box, nodal_values, nhat, l, s, r);
        const auto x = GLL<p>::nodes[r];
        const auto y = LGL<p>::nodes[l];
        const auto z = GLL<p>::nodes[s];
        ASSERT_DOUBLETYPE_NEAR(grad_approx[0], poly.der(0, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[1], poly.der(1, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[2], poly.der(2, x, y, z), my_tol);
      }
    }
  }
  for (int l = 0; l < p; ++l) {
    ArrayND<ftype[p + 1][p + 1]> nhat;
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        nhat(s, r) = interp_scs<p, 2>(nodal_values, l, s, r);
      }
    }
    for (int s = 0; s < p + 1; ++s) {
      for (int r = 0; r < p + 1; ++r) {
        auto grad_approx = gradient_scs<p, 2>(box, nodal_values, nhat, l, s, r);
        const auto x = GLL<p>::nodes[r];
        const auto y = GLL<p>::nodes[s];
        const auto z = LGL<p>::nodes[l];
        ASSERT_DOUBLETYPE_NEAR(grad_approx[0], poly.der(0, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[1], poly.der(1, x, y, z), my_tol);
        ASSERT_DOUBLETYPE_NEAR(grad_approx[2], poly.der(2, x, y, z), my_tol);
      }
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
