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


    double stress_divergence(double x, double y, double z, int d) const
    {
      switch (d)
      {
        case 0:
          return -(kx * kx + ky * ky + kz * kz) * val(x, y, z, XH);
        case 1:
          return -(kx * kx + ky * ky + kz * kz) * val(x, y, z, YH);
        case 2:
          return -(kx * kx + ky * ky + kz * kz) * val(x, y, z, ZH);
        default:
          return 0;
      }
    }

    double advection(double x, double y, double z, int d) const
    {
      switch (d)
      {
        case 0:
          return ((-kx + (kx + ky)*std::cos(2*ky*y) - ky*std::cos(2*kz*z))*std::sin(2*kx*x))/4.;
        case 1:
          return ((-ky + (kx + ky)*std::cos(2*kx*x) - kx*std::cos(2*kz*z))*std::sin(2*ky*y))/4.;
        case 2:
          return ((kx + ky)*(-kx - ky + ky*std::cos(2*kx*x) + kx*std::cos(2*ky*y))*std::sin(2*kz*z))/(4.*kz);
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

        exact_force(k, j, i, XH) = -test.stress_divergence(x, y, z, XH) + test.advection(x,y,z, XH);
        exact_force(k, j, i, YH) = -test.stress_divergence(x, y, z, YH) + test.advection(x,y,z, YH);
        exact_force(k, j, i, ZH) = -test.stress_divergence(x, y, z, ZH) + test.advection(x,y,z, ZH);
      }
    }
  }

  // constants
  nodal_scalar_workview<p, double> work_viscosity(1); auto& viscosity = work_viscosity.view();
  nodal_scalar_workview<p, double> work_rho(1); auto& rho = work_rho.view();
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
TEST_POLY(HexMomentumAdvDiff, mms, 20);
}}

