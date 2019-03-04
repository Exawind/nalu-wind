#include <gtest/gtest.h>
#include <limits>

#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/CVFEMCoefficientMatrices.h>

#include <memory>
#include <tuple>
#include <random>
#include <chrono>

#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

namespace sierra {
namespace nalu {
namespace {

  template <int p> void check_orthogonal()
    {
      nodal_scalar_workview<p, double> l_exactScvVolume(0);
      auto& exactScvVolume = l_exactScvVolume.view();

      std::vector<double> scsLocations1D = gauss_legendre_rule(p).first;
      std::vector<double> paddedScsLocations1D = pad_end_points(scsLocations1D,-1,+1); // add the element ends

      for (int k = 0; k < p+1; ++k) {
        double z_scsL = paddedScsLocations1D[k+0];
        double z_scsR = paddedScsLocations1D[k+1];
        for (int j = 0; j < p+1; ++j) {
          double y_scsL = paddedScsLocations1D[j+0];
          double y_scsR = paddedScsLocations1D[j+1];
          for (int i = 0; i < p+1;++i) {
            double x_scsL = paddedScsLocations1D[i+0];
            double x_scsR = paddedScsLocations1D[i+1];
            exactScvVolume(k, j, i) = (x_scsR - x_scsL) * (y_scsR - y_scsL) * (z_scsR - z_scsL);
          }
        }
      }

      nodal_vector_workview<p, double> l_coords(0);
      auto& coords = l_coords.view();

      double Q[3][3] = {
          { 1, 0, 0 },
          { 0, 1, 0 },
          { 0, 0, 1 }
      };

      std::vector<double> coords1D = gauss_lobatto_legendre_rule(p + 1).first;
      for (int k = 0; k < p + 1; ++k) {
        const double z = coords1D[k];
        for (int j = 0; j < p + 1; ++j) {
          const double y = coords1D[j];
          for (int i = 0; i < p + 1; ++i) {
            const double x = coords1D[i];

            double xvec[3] =
            { Q[0][0] * x + Q[0][1] * y + Q[0][2] * z, Q[1][0] * x + Q[1][1] * y + Q[1][2] * z, Q[2][0] * x + Q[2][1] * y
                + Q[2][2] * z };

            coords(k, j, i, 0) = xvec[0];
            coords(k, j, i, 1) = xvec[1];
            coords(k, j, i, 2) = xvec[2];
          }
        }
      }

      auto ops = CVFEMOperators<p, double>();
      scs_vector_workview<p, double> l_metric(0);
  #ifdef NDEBUG
      int nRuns = 100000 / (p+1);
  #else
      int nRuns = 1;
  #endif

      using clock_type = std::chrono::steady_clock;
      auto start_clock = clock_type::now();
      for (int j = 0; j < nRuns; ++j) {
        high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());
      }
      auto end_metric = clock_type::now();

      const auto& agj = l_metric.view();
      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p; ++i) {
            EXPECT_NEAR(agj(0, k, j, i, 0), -1, 1.0e-12);
            EXPECT_NEAR(agj(0, k, j, i, 1), 0, 1.0e-12);
            EXPECT_NEAR(agj(0, k, j, i, 2), 0, 1.0e-12);
          }
        }
      }

      for (int k = 0; k < p + 1; ++k) {
        for (int j = 0; j < p; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            EXPECT_NEAR(agj(1, k, j, i, 0), 0, 1.0e-12);
            EXPECT_NEAR(agj(1, k, j, i, 1), -1, 1.0e-12);
            EXPECT_NEAR(agj(1, k, j, i, 2), 0, 1.0e-12);
          }
        }
      }

      for (int k = 0; k < p; ++k) {
        for (int j = 0; j < p + 1; ++j) {
          for (int i = 0; i < p + 1; ++i) {
            EXPECT_NEAR(agj(2, k, j, i, 0), 0, 1.0e-12);
            EXPECT_NEAR(agj(2, k, j, i, 1), 0, 1.0e-12);
            EXPECT_NEAR(agj(2, k, j, i, 2), -1, 1.0e-12);
          }
        }
      }


      std::cout << "Over " << nRuns << " runs, avg time for diffusion metric: " <<
          std::chrono::duration_cast<std::chrono::duration<double>>(end_metric - start_clock).count()/(double)nRuns
          << "s" << std::endl;
    }

}

TEST_POLY(DiffusionMetricHO, check_orthogonal, 4)

}
}




