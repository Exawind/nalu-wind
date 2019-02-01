#include <gtest/gtest.h>
#include <limits>

#include <kernel/TensorProductCVFEMDiffusion.h>
#include <kernel/TensorProductCVFEMSource.h>
#include <master_element/TensorProductCVFEMDiffusionMetric.h>
#include <master_element/TensorProductCVFEMVolumeMetric.h>
#include <master_element/CVFEMCoefficientMatrices.h>
#include <Teuchos_BLAS.hpp>
#include <Teuchos_LAPACK.hpp>

#include <memory>
#include <tuple>
#include <random>
#include <chrono>

#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

namespace sierra {
namespace nalu {
namespace {

template <typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
Kokkos::View<typename MatrixTypeA::value_type**>
kron3(std::string label, const MatrixTypeA& A, const MatrixTypeB& B,  const MatrixTypeC& C)
{
  const int nsize = A.extent_int(0);
  const int nsize3 = nsize*nsize*nsize;
  Kokkos::View<typename MatrixTypeA::value_type**> result(label, nsize3, nsize3);

  auto flat_index = [nsize] (int k, int j, int i) { return i + nsize * j  + nsize * nsize * k; };
  for (int n = 0; n < nsize; ++n) {
    for (int m = 0; m < nsize; ++m) {
      for (int l = 0; l < nsize; ++l) {
        for (int k = 0; k < nsize; ++k) {
          for (int j = 0; j < nsize; ++j) {
            for (int i = 0; i < nsize; ++i) {
              result(flat_index(n, m, l), flat_index(k, j, i)) = A(n, k) * B(m, j) * C(l, i);
            }
          }
        }
      }
    }
  }
  return result;
}

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
//--------------------------------------------------------------

template <int p> Kokkos::View<double[(p+1)*(p+1)*(p+1)][(p+1)*(p+1)*(p+1)]>
single_cube_element_laplacian()
{
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);

  auto deriv = coefficients::scs_derivative_weights<p, double>();
  auto integ = coefficients::nodal_integration_weights<p, double>();

  // helper matrices
  auto diff = coefficients::difference_matrix<p, double>();
  auto identity = coefficients::identity_matrix<p, double>();

  Kokkos::View<double[p+1][p+1]> laplacian1D{"laplacian 1D"};
  Kokkos::deep_copy(laplacian1D, 0);

  for (int k = 0; k < p + 1; ++k) {
    laplacian1D(k, 0) = -deriv(0, k);
    for (int j = 1; j < p; ++j) {
      laplacian1D(k, j) = -deriv(j, k) + deriv(j - 1, k);
    }
    laplacian1D(k, p) = +deriv(p - 1, k);
  }

  int ipiv[(p+1)*(p+1)] = {};
  int info = 0;

  auto integ_t = square_transpose(integ); // row v column
  Teuchos::LAPACK<int,double>().GESV(
    p+1, p+1,
    integ_t.data(), p+1,
    ipiv,
    laplacian1D.data(), p+1,
    &info
  );
  EXPECT_EQ(info,0);

  auto dxx = kron3("dxx", laplacian1D, identity, identity);
  auto dyy = kron3("dyy", identity, laplacian1D, identity);
  auto dzz = kron3("dzz", identity, identity, laplacian1D);

  Kokkos::View<double[npe][npe]> laplacian_op{"laplacian_op"};
  for (int j = 0; j < npe; ++j) {
    for (int i = 0; i < npe; ++i) {
      laplacian_op(j, i) = dxx(j, i) + dyy(j, i) + dzz(j, i);
    }
  }
  return laplacian_op;
}

template <int p> Kokkos::View<double[(p+1)*(p+1)*(p+1)][(p+1)*(p+1)*(p+1)]>
single_cube_element_stiffness()
{
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);

  auto deriv = coefficients::scs_derivative_weights<p, double>();
  auto integ = coefficients::nodal_integration_weights<p, double>();

  // helper matrices
  auto diff = coefficients::difference_matrix<p, double>();

  Kokkos::View<double[p+1][p+1]> stiffness{"laplacian 1D"};
  Kokkos::deep_copy(stiffness, 0);

  for (int k = 0; k < p + 1; ++k) {
    stiffness(0, k) = +deriv(0, k);
    for (int j = 1; j < p; ++j) {
      stiffness(j, k) = deriv(j, k) - deriv(j - 1, k);
    }
    stiffness(p, k) = -deriv(p - 1, k);
  }

  auto dxx = kron3("kxx", integ, integ, stiffness);
  auto dyy = kron3("kyy", integ, stiffness, integ);
  auto dzz = kron3("kzz", stiffness, integ, integ);

  Kokkos::View<double[npe][npe]> stiff3d{"stiff_3d"};
  for (int j = 0; j < npe; ++j) {
    for (int i = 0; i < npe; ++i) {
      stiff3d(j, i) = -(dxx(j, i) + dyy(j, i) + dzz(j, i));
    }
  }
  return stiff3d;
}

// must be some periodic function
struct MMSFunction {
  double val(double x, double y, double z) const {
    return (std::cos(c[0] * x) * std::cos(c[1] * y) * std::cos(c[2] * z));
  }

  double laplacian(double x, double y, double z) const {
    return -(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) * val(x, y, z);
  };
  std::array<double, 3> c{{1*M_PI,2*M_PI,1*M_PI}};
};

template <int p> void check_laplacian_coefficients()
{
  MMSFunction mms;

  nodal_scalar_workview<p, double> l_scalar(0);
  auto& scalar = l_scalar.view();

  nodal_scalar_workview<p, double> l_exact_laplacian(0);
  auto& exact_laplacian = l_exact_laplacian.view();

  nodal_vector_workview<p, double> l_coords(0);
  auto& coords = l_coords.view();

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];
        coords(k,j,i,0) = x;
        coords(k,j,i,1) = y;
        coords(k,j,i,2) = z;
        scalar(k,j,i) = mms.val(x,y,z);
        exact_laplacian(k,j,i) = mms.laplacian(x,y,z);
      }
    }
  }

  // Check that our orthogonal element laplacian correctly
  // approximates the Laplacian
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);
  auto laplacian_operator = single_cube_element_laplacian<p>();
  nodal_scalar_workview<p, double> l_numerical_laplacian(0);
  const auto& numerical_laplacian = l_numerical_laplacian.view();
  Teuchos::BLAS<int,double>().GEMV(
    Teuchos::NO_TRANS, // row v column
    npe, npe,
    -1.0,
    laplacian_operator.data(), npe,
    scalar.data(), 1,
    +0.0,
    l_numerical_laplacian.data(), 1
  );
  EXPECT_VIEW_NEAR_3D(exact_laplacian, numerical_laplacian, 1.0e-6); // this one is order 100
}

template <int p>
void check_diffusion_jacobian()
{
  MMSFunction mms;
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);

  nodal_scalar_workview<p, double> l_scalar(0);
  auto& scalar = l_scalar.view();

  nodal_scalar_workview<p, double> l_exact_laplacian(0);
  auto& exact_laplacian = l_exact_laplacian.view();

  nodal_vector_workview<p, double> l_coords(0);
  auto& coords = l_coords.view();

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];
        coords(k,j,i,0) = x;
        coords(k,j,i,1) = y;
        coords(k,j,i,2) = z;
        scalar(k,j,i) = mms.val(x,y,z);
        exact_laplacian(k,j,i) = mms.laplacian(x,y,z);
      }
    }
  }

  auto ops = CVFEMOperators<p, double>();
  scs_vector_workview<p, double> l_metric(0);
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());

  std::vector<double> lhs_data(npe * npe, 0);
  matrix_view<p, double> lhs(lhs_data.data());
  tensor_assembly::scalar_diffusion_lhs(ops, l_metric.view(), lhs);

  auto stiff = single_cube_element_stiffness<p>();
  EXPECT_VIEW_NEAR_2D(lhs, stiff, 1.0e-8);
}


template <int p>
void mms()
{
  MMSFunction mms;

  nodal_scalar_workview<p, double> l_scalar(0);
  auto& scalar = l_scalar.view();

  nodal_scalar_workview<p, double> l_exact_laplacian(0);
  auto& exact_laplacian = l_exact_laplacian.view();

  nodal_vector_workview<p, double> l_coords(0);
  auto& coords = l_coords.view();

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-0.1, 0.1);

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];
        coords(k,j,i,0) = x;
        coords(k,j,i,1) = y;
        coords(k,j,i,2) = z;
        scalar(k,j,i) = mms.val(x,y,z);
        exact_laplacian(k,j,i) = -mms.laplacian(x,y,z);
      }
    }
  }

  auto ops = CVFEMOperators<p, double>();

  nodal_scalar_workview<p, double> work_vol; auto& vol = work_vol.view();
  high_order_metrics::compute_volume_metric_linear(ops, coords, vol);

  nodal_scalar_workview<p, double> work_rhs(0); auto& rhs = work_rhs.view();

  // compute source from exact residual:
  tensor_assembly::add_volumetric_source(ops, vol, exact_laplacian, rhs);

  scs_vector_workview<p, double> l_metric(0);
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());

  tensor_assembly::scalar_diffusion_rhs(ops, l_metric.view(), scalar, rhs);

  for (int k = 1; k < p; ++k) {
    for (int j = 1; j < p; ++j) {
      for (int i = 1; i < p; ++i) {
        ASSERT_NEAR(0, rhs(k,j,i), 1.0e-8)  << "k,j,i: " << k << ", " << j << ", " << i;
      }
    }
  }
}


template <int p>
void check_diffusion_jacobian_is_consistent()
{
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);

  nodal_scalar_workview<p, double> l_scalar(0);
  auto& scalar = l_scalar.view();

  nodal_vector_workview<p, double> l_coords(0);
  auto& coords = l_coords.view();

  double Q[3][3] = {
      {1,0,1},{0,1,-1},{0.5,0.5,1}
  };
  ASSERT_TRUE(determinant33(&Q[0][0]) > 0);

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-1, 1);

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];

        double xvec[3] = {
            Q[0][0] * x + Q[0][1] * y + Q[0][2] * z,
            Q[1][0] * x + Q[1][1] * y + Q[1][2] * z,
            Q[2][0] * x + Q[2][1] * y + Q[2][2] * z
        };

        coords(k, j, i, 0) = xvec[0];
        coords(k, j, i, 1) = xvec[1];
        coords(k, j, i, 2) = xvec[2];

        scalar(k,j,i) = coeff(rng);
      }
    }
  }

  auto ops = CVFEMOperators<p, double>();
  scs_vector_workview<p, double> l_metric(0);
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());

  std::vector<double> lhs_data(npe * npe, 0);
  matrix_view<p, double> lhs(lhs_data.data());
  tensor_assembly::scalar_diffusion_lhs(ops, l_metric.view(), lhs);

  nodal_scalar_workview<p, double> l_rhs(0);
  Teuchos::BLAS<int,double>().GEMV(
    Teuchos::TRANS, // row v column
    npe, npe,
    -1.0,
    lhs.data(), npe,
    scalar.data(), 1,
    +1.0,
    l_rhs.data(), 1
  );

  nodal_scalar_workview<p, double> l_rhs_jf(0);
  tensor_assembly::scalar_diffusion_rhs(ops, l_metric.view(), scalar, l_rhs_jf.view());
  EXPECT_VIEW_NEAR_3D(l_rhs.view(), l_rhs_jf.view(), 1.0e-8);
}

template <int p>
void laplacian_jacobian_timing()
{
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);

  nodal_vector_workview<p, DoubleType> l_coords(0);
  auto& coords = l_coords.view();

  double Q[3][3] = {
      {1,0,1},{0,1,-1},{0.5,0.5,1}
  };
  ASSERT_TRUE(determinant33(&Q[0][0]) > 0);


  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];

        double xvec[3] = {
            Q[0][0] * x + Q[0][1] * y + Q[0][2] * z,
            Q[1][0] * x + Q[1][1] * y + Q[1][2] * z,
            Q[2][0] * x + Q[2][1] * y + Q[2][2] * z
        };

        coords(k, j, i, 0) = xvec[0];
        coords(k, j, i, 1) = xvec[1];
        coords(k, j, i, 2) = xvec[2];
      }
    }
  }

  auto ops = CVFEMOperators<p, DoubleType>();
  scs_vector_workview<p, DoubleType> l_metric(0);
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());

  AlignedVector<DoubleType> lhs_data(npe * npe, 0);
  matrix_view<p, DoubleType> lhs(lhs_data.data());

  int nRuns = 8e4/((p+1)*(p+1)*(p+1));
  using clock_type = std::chrono::steady_clock;
  auto start_clock = clock_type::now();
  for (int n = 0; n < nRuns; ++n) {
    tensor_assembly::scalar_diffusion_lhs(ops, l_metric.view(), lhs);
  }
  auto end_clock =  clock_type::now();

  double avg_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_clock - start_clock).count()/(double)nRuns;
  double avg_time_per_node = avg_time / npe;
  std::cout << "Over " << nRuns << " runs, avg time for Laplacian element matrix assembly: " <<
      avg_time << "s, (" <<  avg_time_per_node << "s per node) for order " << std::to_string(p) << std::endl;
}

template <int p>
void laplacian_residual_timing()
{
  MMSFunction mms;
  constexpr int npe = (p + 1) * (p + 1) * (p + 1);

  nodal_vector_workview<p, DoubleType> l_coords(0);
  auto& coords = l_coords.view();

  double Q[3][3] = {
      {1,0,1},{0,1,-1},{0.5,0.5,1}
  };
  ASSERT_TRUE(determinant33(&Q[0][0]) > 0);

  nodal_scalar_workview<p, DoubleType> l_scalar(0);
  auto& scalar = l_scalar.view();

  std::vector<double> coords1D = gauss_lobatto_legendre_rule(p+1).first;
  for (int k = 0; k < p + 1; ++k) {
    const double z = coords1D[k];
    for (int j = 0; j < p+1; ++j) {
      const double y = coords1D[j];
      for (int i = 0; i < p+1;++i) {
        const double x = coords1D[i];

        double xvec[3] = {
            Q[0][0] * x + Q[0][1] * y + Q[0][2] * z,
            Q[1][0] * x + Q[1][1] * y + Q[1][2] * z,
            Q[2][0] * x + Q[2][1] * y + Q[2][2] * z
        };

        coords(k, j, i, 0) = xvec[0];
        coords(k, j, i, 1) = xvec[1];
        coords(k, j, i, 2) = xvec[2];

        scalar(k, j, i) = mms.val(xvec[0], xvec[1], xvec[2]);
      }
    }
  }

  auto ops = CVFEMOperators<p, DoubleType>();
  scs_vector_workview<p, DoubleType> l_metric(0);
  auto& metric = l_metric.view();
  high_order_metrics::compute_laplacian_metric_linear(ops, coords, l_metric.view());

  nodal_scalar_workview<p, DoubleType> l_rhs_jf(0);
  auto& rhs = l_rhs_jf.view();

  const int nRuns = 8e5/((p+1)*(p+1)*(p+1));
  using clock_type = std::chrono::steady_clock;
  auto start_clock = clock_type::now();
  for (int n = 0; n < nRuns; ++n) {
    tensor_assembly::scalar_diffusion_rhs(ops, metric, scalar, rhs);
  }
  auto end_clock =  clock_type::now();

  double avg_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_clock - start_clock).count()/(double)nRuns;
  double avg_time_per_node = avg_time / npe;
  std::cout << "Over " << nRuns << " runs, avg time for Laplacian single element residual assembly: " <<
      avg_time << "s, (" <<  avg_time_per_node << "s per node) for order " << std::to_string(p) << std::endl;
}

}
//--------------------------------------------------------------
TEST_POLY(HexDiffusion, check_diffusion_jacobian_is_consistent, 4)
TEST_POLY(HexDiffusion, check_diffusion_jacobian, 4)
TEST_POLY(HexDiffusion, mms, 20)

#ifndef DNDEBUG
TEST_POLY(HexDiffusion, check_laplacian_coefficients, 18)
TEST_POLY_to5(HexDiffusion, laplacian_jacobian_timing)
TEST_POLY_to5(HexDiffusion, laplacian_residual_timing)
#endif

}}
