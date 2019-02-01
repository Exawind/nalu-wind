#include <gtest/gtest.h>
#include <limits>

#include <kernel/TensorProductCVFEMScalarAdvDiff.h>
#include <kernel/TensorProductCVFEMSource.h>

#include <master_element/TensorProductCVFEMAdvectionMetric.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>


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


struct TestFunction
{
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

  double val(double x, double y, double z) const
  {
    return std::cos(2 * kx * x) * std::cos(ky * y) * std::cos(kz * z);
  }

  double gamma(double x, double y, double z) const {
    return (2 + std::cos(kx * x) * std::cos(2 * ky * y) * std::cos(kz * z));
  }

  double rho(double x, double y, double z) const
  {
    return (2 + std::sin(kx * x) * std::sin(ky * y) * std::sin(2 * kz * z));
  }

  double residual(double x, double y, double z) const {
    return -(std::cos(2*kx*x)*(2*std::cos(ky*y)*(-2*(4*std::pow(kx,2) + std::pow(ky,2) + std::pow(kz,2))*std::cos(kz*z) +
        std::pow(std::cos(kz*z),2)*(std::cos(kx*x)*(2*std::pow(ky,2) - (4*std::pow(kx,2) + 3*std::pow(ky,2) + std::pow(kz,2))*std::cos(2*ky*y)) +
           2*(kx + ky)*std::cos(2*kz*z)*std::pow(std::sin(kx*x),2)*std::pow(std::sin(ky*y),2)) + std::pow(kz,2)*std::cos(kx*x)*std::cos(2*ky*y)*std::pow(std::sin(kz*z),2)) -
     kx*std::sin(kx*x)*std::sin(2*ky*y)*std::sin(2*kz*z) - std::cos(ky*y)*(ky*std::pow(std::cos(ky*y),2)*std::pow(std::sin(kx*x),2) + kx*std::pow(std::sin(ky*y),2))*std::pow(std::sin(2*kz*z),2)))/2. -
kx*std::sin(kx*x)*std::sin(2*kx*x)*(1.0/std::tan(kx*x)*std::sin(2*ky*y)*std::sin(2*kz*z) +
  std::cos(ky*y)*(2*kx*std::cos(2*ky*y)*std::pow(std::cos(kz*z),2) + std::cos(kx*x)*std::pow(std::sin(ky*y),2)*std::pow(std::sin(2*kz*z),2)));
  }

  const double kx{ M_PI };
  const double ky{ M_PI };
  const double kz{ M_PI };
};

template <int p> void mms()
{
  // check that we get the correct answer given a single, very high p element
  nodal_vector_workview<p, double> work_coords; auto& coords = work_coords.view();
  nodal_vector_workview<p, double> work_velocity; auto& velocity = work_velocity.view();
  nodal_scalar_workview<p, double> work_scalar; auto& scalar = work_scalar.view();

  // constants
  nodal_scalar_workview<p, double> work_diffusivity(1); auto& diffusivity = work_diffusivity.view();
  nodal_scalar_workview<p, double> work_rho(1); auto& rho = work_rho.view();
  nodal_vector_workview<p, double> work_Gp(0); auto& Gp = work_Gp.view();
  nodal_scalar_workview<p, double> work_pressure(1); auto& pressure = work_pressure.view();

  TestFunction test;
  nodal_scalar_workview<p, double> work_exact_resid(0); auto& exact_resid = work_exact_resid.view();

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];
        coords(k, j, i, XH) = x;
        coords(k, j, i, YH) = y;
        coords(k, j, i, ZH) = z;

        rho(k, j, i) = test.rho(x, y, z);
        diffusivity(k, j, i) = test.gamma(x, y, z);

        velocity(k, j, i, XH) = test.uval(x, y, z, XH);
        velocity(k, j, i, YH) = test.uval(x, y, z, YH);
        velocity(k, j, i, ZH) = test.uval(x, y, z, ZH);

        scalar(k,j,i) = test.val(x,y,z);

        exact_resid(k, j, i) = test.residual(x,y,z);
      }
    }
  }

  // volume integrate source
  CVFEMOperators<p, double> ops;

  nodal_scalar_workview<p, double> work_vol; auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops, coords, vol);

  nodal_scalar_workview<p, double> work_rhs(0); auto& rhs = work_rhs.view();

  // compute source from exact residual:
  tensor_assembly::add_volumetric_source(ops, vol, exact_resid, rhs);

  // compute numerical residual:
  scs_vector_workview<p, double> work_diffmetric; auto& diffmetric = work_diffmetric.view();
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, diffmetric);

  scs_scalar_workview<p, double> l_mdot(0); auto& mdot = l_mdot.view();
  high_order_metrics::compute_mdot_linear(ops, coords, diffmetric, 1,  rho, velocity, Gp, pressure, mdot);

  high_order_metrics::scale_metric(ops, diffusivity, diffmetric);
  tensor_assembly::scalar_advdiff_rhs(ops, mdot, diffmetric, scalar, rhs);

  // consider just element interior
  for (int k = 1; k < p; ++k) {
    for (int j = 1; j < p; ++j) {
      for (int i = 1; i < p; ++i) {
        EXPECT_NEAR(0, rhs(k,j,i), 1.0e-8) << "k,j,i: " << k << ", " << j << ", " << i;
      }
    }
  }
}


}
//--------------------------------------------------------------
TEST_POLY(ScalarAdvDiff, mms, 22)
}}
