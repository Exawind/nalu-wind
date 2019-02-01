#include <gtest/gtest.h>
#include <limits>

#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <master_element/CVFEMCoefficientMatrices.h>


#include <memory>
#include <tuple>
#include <random>
#include <chrono>

#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"



namespace sierra {
namespace nalu {

template <int p> void check_scv_volumes()
{
  nodal_scalar_workview<p, double> work_exactScvVolume(0);
  auto& exactScvVolume = work_exactScvVolume.view();

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

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;

  for (int k = 0; k < p + 1; ++k) {
    for (int j = 0; j < p + 1; ++j) {
      for (int i = 0; i < p + 1; ++i) {
        coords(k,j,i,0) = coords1D[i];
        coords(k,j,i,1) = coords1D[j];
        coords(k,j,i,2) = coords1D[k];
      }
    }
  }

  auto ops = CVFEMOperators<p, double>();
  nodal_scalar_workview<p, double> l_detj(0);
  nodal_scalar_workview<p, double> l_computedScvVolume(0);

#ifdef NDEBUG
  int nRuns = 100000 / (p+1);
#else
  int nRuns = 1;
#endif

  using clock_type = std::chrono::steady_clock;
  auto start_clock = clock_type::now();
  for (int j = 0; j < nRuns; ++j) {
    Kokkos::deep_copy(l_detj.view(),0);
    high_order_metrics::compute_volume_metric_linear(ops, coords, l_detj.view());
  }
  auto end_metric = clock_type::now();
  for (int j = 0; j < nRuns; ++j) {
    Kokkos::deep_copy(l_computedScvVolume.view(),0);
    ops.volume(l_detj.view(), l_computedScvVolume.view());
  }
  auto end_volume = clock_type::now();
  EXPECT_VIEW_NEAR_3D(exactScvVolume, l_computedScvVolume.view(), 1.0e-10);

  std::cout << "Over " << nRuns << " runs, avg time for volume metric: " << std::chrono::duration_cast<std::chrono::duration<double>>(end_metric - start_clock).count()/(double)nRuns
     << "s | avg time for volume integration: " <<  std::chrono::duration_cast<std::chrono::duration<double>>(end_volume - end_metric).count()/(double)nRuns <<  "s" << std::endl;

}

TEST_POLY(VolumeHO, check_scv_volumes, 4)

}
}




