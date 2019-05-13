#include <gtest/gtest.h>
#include <limits>

#include <kernel/TensorProductCVFEMMomentumAdvDiff.h>
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


  struct TestFunction {
    double val(double x, double y, double z, int d) const
    {
      const double zcoeff = (kz != 0 ) ? -(kx + ky)/kz : 0;
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

    double mu(double x, double y, double z) { return (2 + std::cos(kx *x) * std::cos(ky*y) * std::cos(kz*z)); }
    double rho(double x, double y, double z) { return (2 + std::sin(kx *x) * std::sin(ky*y) * std::sin(kz*z)); }

    double force(double x, double y, double z, int d) {
      ThrowRequire(d >= 0 && d <= 2);
      switch(d) {
        case XH:
          return 2*(std::pow(kx,2) + std::pow(ky,2) + std::pow(kz,2))*std::cos(kx*x)*std::sin(ky*y)*std::sin(kz*z) -
              std::sin(2*kx*x)*(-(ky*std::pow(std::cos(ky*y),2)) + kx*std::pow(std::sin(ky*y),2))*std::pow(std::sin(kz*z),2) +
              (std::sin(ky*y)*(4*ky*std::cos(kx*x)*std::pow(std::cos(ky*y),2)*std::pow(std::sin(kx*x),2) + kx*(std::cos(kx*x) + std::cos(3*kx*x))*std::pow(std::sin(ky*y),2))*std::pow(std::sin(kz*z),3))/2. -
              (kx + ky)*std::pow(std::cos(kz*z),2)*std::sin(2*kx*x)*std::pow(std::sin(ky*y),2)*(1 + std::sin(kx*x)*std::sin(ky*y)*std::sin(kz*z)) +
              (((std::pow(ky,2) + std::pow(kz,2))*std::pow(std::cos(kx*x),2) - std::pow(kx,2)*std::pow(std::sin(kx*x),2))*std::sin(2*ky*y)*std::sin(2*kz*z))/2.;

        case YH:
         return -((kx + ky)*std::pow(std::cos(kz*z),2)*std::pow(std::sin(kx*x),2)*std::sin(2*ky*y)*(1 + std::sin(kx*x)*std::sin(ky*y)*std::sin(kz*z))) +
             (-(std::sin(kz*z)*(-2*ky*std::pow(std::cos(ky*y),3)*std::pow(std::sin(kx*x),3)*std::pow(std::sin(kz*z),2) -
                     4*std::cos(ky*y)*std::sin(kx*x)*(std::pow(kx,2) + std::pow(ky,2) + std::pow(kz,2) + kx*std::pow(std::cos(kx*x),2)*std::pow(std::sin(ky*y),2)*std::pow(std::sin(kz*z),2)) +
                     std::sin(2*ky*y)*std::sin(kz*z)*(-2*kx*std::pow(std::cos(kx*x),2) + ky*std::pow(std::sin(kx*x),2)*(2 + std::sin(kx*x)*std::sin(ky*y)*std::sin(kz*z))))) +
                std::sin(2*kx*x)*((std::pow(kx,2) + std::pow(kz,2))*std::pow(std::cos(ky*y),2) - std::pow(ky,2)*std::pow(std::sin(ky*y),2))*std::sin(2*kz*z))/2.;
        case ZH:
          return -((kx + ky)*(-2*(kx + ky)*std::pow(std::cos(kz*z),3)*std::pow(std::sin(kx*x),3)*std::pow(std::sin(ky*y),3) + (std::pow(kx,2) + std::pow(ky,2))*std::pow(std::cos(kz*z),2)*std::sin(2*kx*x)*std::sin(2*ky*y) -
                  std::pow(kz,2)*std::sin(2*kx*x)*std::sin(2*ky*y)*std::pow(std::sin(kz*z),2) +
                  4*std::cos(kz*z)*std::sin(kx*x)*std::sin(ky*y)*(std::pow(kx,2) + std::pow(ky,2) + std::pow(kz,2) +
                     (ky*std::pow(std::cos(ky*y),2)*std::pow(std::sin(kx*x),2) + kx*std::pow(std::cos(kx*x),2)*std::pow(std::sin(ky*y),2))*std::pow(std::sin(kz*z),2)) +
                  (kx + ky - ky*std::cos(2*kx*x) - kx*std::cos(2*ky*y) + (kx + ky)*std::pow(std::sin(kx*x),3)*std::pow(std::sin(ky*y),3)*std::sin(kz*z))*std::sin(2*kz*z)))/(2.*kz);
        default:
          return 0;

      }
    }

    double kx{M_PI};
    double ky{M_PI};
    double kz{M_PI};
  };


template <int p> void mms()
{
  // check that we converge given a single, very high p element

  nodal_vector_workview<p, double> l_velocity; auto& velocity = l_velocity.view();
  nodal_vector_workview<p, double> l_coords; auto& coords = l_coords.view();

  TestFunction test;
  nodal_vector_workview<p, double> l_exact_force(0); auto& exact_force = l_exact_force.view();

  nodal_scalar_workview<p, double> work_viscosity(1); auto& viscosity = work_viscosity.view();
  nodal_scalar_workview<p, double> work_rho(1); auto& rho = work_rho.view();

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

        velocity(k, j, i, XH) = test.val(x, y, z, XH);
        velocity(k, j, i, YH) = test.val(x, y, z, YH);
        velocity(k, j, i, ZH) = test.val(x, y, z, ZH);

        viscosity(k,j,i) = test.mu(x,y,z);
        rho(k,j,i) = test.rho(x,y,z);

        exact_force(k, j, i, XH) = test.force(x, y, z, XH);
        exact_force(k, j, i, YH) = test.force(x, y, z, YH);
        exact_force(k, j, i, ZH) = test.force(x, y, z, ZH);
      }
    }
  }

  // constants
  nodal_vector_workview<p, double> work_Gp(0); auto& Gp = work_Gp.view();
  nodal_scalar_workview<p, double> work_pressure(1); auto& pressure = work_pressure.view();

  // volume integrate source
  CVFEMOperators<p, double> ops;

  nodal_scalar_workview<p, double> work_vol; auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops, coords, vol);

  nodal_vector_workview<p, double> work_rhs(0); auto& rhs = work_rhs.view();

  // compute source from exact residual:
  tensor_assembly::add_volumetric_source(ops, vol, exact_force, rhs);

  // compute numerical residual:
  scs_vector_workview<p, double> work_lapmetric; auto& lapmetric = work_lapmetric.view();
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, lapmetric);

  scs_scalar_workview<p, double> work_mdot(0); auto& mdot = work_mdot.view();
  high_order_metrics::compute_mdot_linear(ops, coords, lapmetric, 1,  rho, velocity, Gp, pressure, mdot);

  scs_vector_workview<p, double> work_tau_dot_a(0); auto& tau_dot_a = work_tau_dot_a.view();
  tensor_assembly::area_weighted_face_normal_shear_stress(ops, coords, viscosity, velocity, tau_dot_a);

  tensor_assembly::momentum_advdiff_rhs(ops, tau_dot_a, mdot, velocity, rhs);

  // consider just element interior
  for (int k = 1; k < p; ++k) {
    for (int j = 1; j < p; ++j) {
      for (int i = 1; i < p; ++i) {
        EXPECT_NEAR(0, rhs(k,j,i, XH), 1.0e-8) << "x | k,j,i: " << k << ", " << j << ", " << i;
        EXPECT_NEAR(0, rhs(k,j,i, YH), 1.0e-8) << "y | k,j,i: " << k << ", " << j << ", " << i;
        EXPECT_NEAR(0, rhs(k,j,i, ZH), 1.0e-8) << "z | k,j,i: " << k << ", " << j << ", " << i;
      }
    }
  }
}

}
//--------------------------------------------------------------
TEST_POLY(HexMomentumAdvDiff, mms, 20)
}}

