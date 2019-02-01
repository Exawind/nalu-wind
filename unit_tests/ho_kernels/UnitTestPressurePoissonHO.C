#include <gtest/gtest.h>
#include <limits>

#include <kernel/TensorProductCVFEMPressurePoisson.h>
#include <kernel/TensorProductCVFEMSource.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMAdvectionMetric.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>



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


namespace sierra{
namespace nalu {
namespace {

  struct TestFunction {
    double uval(double x, double y, double z, int d) const
    {
      const double zcoeff = (kz != 0) ? -(kx + ky) / kz : 0;
      switch (d)
      {
        case 0:
          return std::cos(kx * x) * std::sin(ky * y) * std::sin(kz * z);
        case 1:
          return std::sin(kx * x) * std::cos(ky * y) * std::sin(kz * z);
        case 2:
          return zcoeff * std::sin(kx * x) * std::sin(ky * y) * std::cos(kz * z);
        default:
          return 0;
      }
    }

    double rho(double x, double y, double z) const
    {
      return (2 + std::sin(kx * x) * std::sin(ky * y) * std::sin(2 * kz * z));
    }

    double residual(double x, double y, double z) const
    {
      return -2*(kx + ky)*std::cos(kz*z)*std::cos(2*kz*z)*std::pow(std::sin(kx*x),2)*std::pow(std::sin(ky*y),2)
      + (ky*std::pow(std::cos(ky*y),2)*std::pow(std::sin(kx*x),2) + kx*std::pow(std::cos(kx*x),2)*std::pow(std::sin(ky*y),2))*std::sin(kz*z)*std::sin(2*kz*z);
    }

    const double kx{M_PI};
    const double ky{M_PI};
    const double kz{M_PI};
  };

  template <int p> void mms()
  {
    TestFunction test;

    nodal_vector_workview<p, double> work_velocity; auto& velocity = work_velocity.view();
    nodal_scalar_workview<p, double> work_density; auto& density = work_density.view();

    nodal_scalar_workview<p, double> work_pressure(1); auto& pressure = work_pressure.view();
    nodal_vector_workview<p, double> work_Gp(0); auto& Gp = work_Gp.view();

    nodal_vector_workview<p, double> work_coords; auto& coords = work_coords.view();
    nodal_scalar_workview<p, double> work_exact_resid; auto& exact_resid = work_exact_resid.view();

    std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
    for (int k = 0; k < p + 1; ++k) {
      const double z = coords1D[k];
      for (int j = 0; j < p+1; ++j) {
        const double y = coords1D[j];
        for (int i = 0; i < p+1;++i) {
          const double x = coords1D[i];
          coords(k, j, i, 0) = x;
          coords(k, j, i, 1) = y;
          coords(k, j, i, 2) = z;


          velocity(k, j, i, 0) = test.uval(x, y, z, XH);
          velocity(k, j, i, 1) = test.uval(x, y, z, YH);
          velocity(k, j, i, 2) = test.uval(x, y, z, ZH);

          density(k, j, i) = test.rho(x,y,z);

          exact_resid(k, j, i) = test.residual(x, y, z);
        }
      }
    }

    auto ops = CVFEMOperators<p, double>();

    nodal_scalar_workview<p, double> work_vol; auto& vol = work_vol.view();
    high_order_metrics::compute_volume_metric_linear(ops, coords, vol);

    nodal_scalar_workview<p, double> work_rhs(0); auto& rhs = work_rhs.view();

    // compute source from exact residual:
    tensor_assembly::add_volumetric_source(ops, vol, exact_resid, rhs);

    scs_vector_workview<p, double> work_metric(0);
    high_order_metrics::compute_laplacian_metric_linear(ops, coords, work_metric.view());

    scs_scalar_workview<p, double> work_mdot(0);
    high_order_metrics::compute_mdot_linear(ops, coords, work_metric.view(), 1.0, density, velocity, Gp, pressure, work_mdot.view());

    tensor_assembly::pressure_poisson_rhs(ops, 1.0, work_mdot.view(), rhs);

    for (int k = 1; k < p; ++k) {
      for (int j = 1; j < p; ++j) {
        for (int i = 1; i < p; ++i) {
          EXPECT_NEAR(rhs(k,j,i), 0, 1.0e-8) << "(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        }
      }
    }
  }
}
//--------------------------------------------------------------
TEST_POLY(PressurePoissonHex, mms, 22)
}}
