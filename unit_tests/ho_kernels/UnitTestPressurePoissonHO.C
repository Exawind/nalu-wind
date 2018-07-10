#include <gtest/gtest.h>
#include <limits>

#include <kernel/TensorProductCVFEMPressurePoisson.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <Kokkos_Core.hpp>
#include <Teuchos_BLAS.hpp>
#include <Teuchos_LAPACK.hpp>

#include <memory>
#include <tuple>
#include <random>
#include <chrono>

#include <element_promotion/ElementDescription.h>
#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

namespace {


  //--------------------------------------------------------------
  template<typename T>
  T square_transpose(const T A)
  {
    T At(A.label()+"_t");

    EXPECT_EQ(A.extent(0), A.extent(1));
    const int n = A.extent_int(0);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        At(i,j) = A(j,i);
      }
    }
    return At;
  }

  struct TestFunction {
    double val(int d, double x, double y, double z) const
    {
      const double zcoeff = (c[2] != 0 ) ? -(c[0] + c[1])/c[2] : 0;
      switch (d)
      {
        case 0:
         return std::cos(c[0] * x) * std::sin(c[1] * y) * std::sin(c[2] * z);
        case 1:
          return std::sin(c[0] * x) * std::cos(c[1] * y) * std::sin(c[2] * z);
        case 2:
          return zcoeff * std::sin(c[0] * x) * std::sin(c[1] * y) * std::cos(c[2] * z);
        default:
          return 0;
      }
    }
    std::array<double, 3> c{{1*M_PI,1*M_PI,1*M_PI}};
  };

  template <int p> void mms()
  {
    TestFunction test;

    sierra::nalu::nodal_vector_workview<p, double> l_velocity; auto& velocity = l_velocity.view();
    sierra::nalu::nodal_scalar_workview<p, double> l_pressure; auto& pressure = l_pressure.view();
    sierra::nalu::nodal_scalar_workview<p, double> l_density; auto& density = l_pressure.view();
    sierra::nalu::nodal_vector_workview<p, double> l_Gp; auto& Gp = l_Gp.view();

    typename sierra::nalu::nodal_vector_workview<p, double> l_coords; auto& coords = l_coords.view();

    std::vector<double> coords1D = sierra::nalu::gauss_lobatto_legendre_rule(p+1).first;
    for (int k = 0; k < p + 1; ++k) {
      const double z = coords1D[k];
      for (int j = 0; j < p+1; ++j) {
        const double y = coords1D[j];
        for (int i = 0; i < p+1;++i) {
          const double x = coords1D[i];
          coords(k, j, i, 0) = x;
          coords(k, j, i, 1) = y;
          coords(k, j, i, 2) = z;
          Gp(k, j, i, 0) = 0;
          Gp(k, j, i, 1) = 0;
          Gp(k, j, i, 2) = 0;

          velocity(k, j, i, 0) = test.val(0, x, y, z);
          velocity(k, j, i, 1) = test.val(1, x, y, z);
          velocity(k, j, i, 2) = test.val(2, x, y, z);

          pressure(k, j, i) = 0;
          density(k, j, i) = 1;

        }
      }
    }

    auto ops = sierra::nalu::CVFEMOperators<p, double>();

    sierra::nalu::scs_vector_workview<p, double> l_metric(0);
    sierra::nalu::high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());

    sierra::nalu::scs_scalar_workview<p, double> l_mdot(0);
    sierra::nalu::high_order_metrics::compute_mdot_linear(ops, coords, l_metric.view(), 1.0, density, velocity, Gp, pressure, l_mdot.view());

    sierra::nalu::nodal_scalar_workview<p, double> l_rhs(0);
    auto& rhs = l_rhs.view();
    sierra::nalu::tensor_assembly::pressure_poisson_rhs(ops, 1.0, l_mdot.view(), rhs);

    for (int k = 1; k < p; ++k) {
      for (int j = 1; j < p; ++j) {
        for (int i = 1; i < p; ++i) {
         ASSERT_NEAR(rhs(k,j,i), 0, 1.0e-8) << "(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        }
      }
    }
  }
}
//--------------------------------------------------------------
TEST_POLY(PressurePoissonHex, mms, 20);
