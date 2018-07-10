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
  double val(double x, double y, double z) const
  {
    return std::cos(2 * kx * x) * std::cos(ky * y) * std::cos(kz * z);
  }

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

  double laplacian(double x, double y, double z) const
  {
    return -(4 * kx * kx + ky * ky + kz * kz) * val(x, y, z);
  }

  double advection(double x, double y, double z) const
  {
    return -(kx * (3 * std::sin(kx * x) + std::sin(3 * kx * x)) * std::sin(2 * ky * y) * std::sin(2 * kz * z)) / 8.;
  }

  const double kx{ M_PI };
  const double ky{ M_PI };
  const double kz{ M_PI };
};


template <int p> void mms()
{
  // check that we converge given a single, very high p element
  sierra::nalu::nodal_vector_workview<p, double> work_coords; auto& coords = work_coords.view();
  sierra::nalu::nodal_vector_workview<p, double> work_velocity; auto& velocity = work_velocity.view();
  sierra::nalu::nodal_scalar_workview<p, double> work_scalar; auto& scalar = work_scalar.view();

  // constants
  sierra::nalu::nodal_scalar_workview<p, double> work_diffusivity(1); auto& diffusivity = work_diffusivity.view();
  sierra::nalu::nodal_scalar_workview<p, double> work_rho(1); auto& rho = work_rho.view();
  sierra::nalu::nodal_vector_workview<p, double> work_Gp(0); auto& Gp = work_Gp.view();
  sierra::nalu::nodal_scalar_workview<p, double> work_pressure(1); auto& pressure = work_pressure.view();

  TestFunction test;
  sierra::nalu::nodal_scalar_workview<p, double> work_exact_resid(0); auto& exact_resid = work_exact_resid.view();

  std::vector<double> coords1D = sierra::nalu::gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];
        coords(k, j, i, XH) = x;
        coords(k, j, i, YH) = y;
        coords(k, j, i, ZH) = z;

        velocity(k, j, i, XH) = test.uval(x, y, z, XH);
        velocity(k, j, i, YH) = test.uval(x, y, z, YH);
        velocity(k, j, i, ZH) = test.uval(x, y, z, ZH);

        scalar(k,j,i) = test.val(x,y,z);

        exact_resid(k, j, i) = -test.laplacian(x, y, z) + test.advection(x,y,z);
      }
    }
  }

  // volume integrate source
  sierra::nalu::CVFEMOperators<p, double> ops;

  sierra::nalu::nodal_scalar_workview<p, double> work_vol; auto& vol = work_vol.view();
  sierra::nalu::high_order_metrics::compute_volume_metric_linear(ops, coords, vol);

  sierra::nalu::nodal_scalar_workview<p, double> work_rhs(0); auto& rhs = work_rhs.view();

  // compute source from exact residual:
  sierra::nalu::tensor_assembly::add_volumetric_source(ops, vol, exact_resid, rhs);

  // compute numerical residual:
  sierra::nalu::scs_vector_workview<p, double> work_diffmetric; auto& diffmetric = work_diffmetric.view();
  sierra::nalu::high_order_metrics::compute_laplacian_metric_linear(ops, coords, diffmetric);

  sierra::nalu::scs_scalar_workview<p, double> l_mdot(0); auto& mdot = l_mdot.view();
  sierra::nalu::high_order_metrics::compute_mdot_linear(ops, coords, diffmetric, 1,  rho, velocity, Gp, pressure, mdot);

  sierra::nalu::high_order_metrics::scale_metric(ops, diffusivity, diffmetric);
  sierra::nalu::tensor_assembly::scalar_advdiff_rhs(ops, mdot, diffmetric, scalar, rhs);

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
TEST_POLY(ScalarAdvDiff, mms, 20);
}}
