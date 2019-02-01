#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <element_promotion/ElementDescription.h>
#include <CVFEMTypeDefs.h>
#include <master_element/TensorProductCVFEMOperators.h>

#include "UnitTestViewUtils.h"
#include "UnitTestUtils.h"

namespace {
  double my_tol = 1.0e-10;

double poly_val(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j);
  }
  return val;
}
//--------------------------------------------------------------------------
double poly_der(std::vector<double> coeffs, double x)
{
  double val = 0.0;
  for (unsigned j = 1; j < coeffs.size(); ++j) {
    val += coeffs[j]*std::pow(x,j-1)*j;
  }
  return val;
}
//--------------------------------------------------------------------------
double poly_int(std::vector<double> coeffs,
  double xlower, double xupper)
{
  double upper = 0.0; double lower = 0.0;
  for (unsigned j = 0; j < coeffs.size(); ++j) {
    upper += coeffs[j]*std::pow(xupper,j+1)/(j+1.0);
    lower += coeffs[j]*std::pow(xlower,j+1)/(j+1.0);
  }
  return (upper-lower);
}

struct TensorPoly {
  TensorPoly(int p) {
    std::mt19937 rng;
    rng.seed(0);
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);
    for (int j = 0; j < p + 1; ++j) {
      coeffsX.push_back(coeff(rng));
      coeffsY.push_back(coeff(rng));
      coeffsZ.push_back(coeff(rng));
    }
  }

  double operator()(double x, double y, double z) {
    return poly_val(coeffsX, x) * poly_val(coeffsY, y) * poly_val(coeffsZ, z);
  }

  double grad_x(double x, double y, double z)  {
    return poly_der(coeffsX,x) * poly_val(coeffsY, y) * poly_val(coeffsZ, z);
  }

  double grad_y(double x, double y, double z) {
    return poly_val(coeffsX,x) * poly_der(coeffsY, y) * poly_val(coeffsZ, z);
  }

  double grad_z(double x, double y, double z) {
    return poly_val(coeffsX,x) * poly_val(coeffsY, y) * poly_der(coeffsZ, z);
  }

  double xyz_integral(double xl, double xr, double yl, double yr, double zl, double zr)  {
    return poly_int(coeffsX,xl,xr) * poly_int(coeffsY, yl, yr) * poly_int(coeffsZ, zl, zr);
  }
  std::vector<double> coeffsX;
  std::vector<double> coeffsY;
  std::vector<double> coeffsZ;
};


//--------------------------------------------------------------------------
template <int p>
void scs_interp_hex()
{
   auto elem = sierra::nalu::ElementDescription::create(3, p);
  auto ops = sierra::nalu::CVFEMOperators<p, double>();

  sierra::nalu::nodal_scalar_workview<p, double> l_nodalValues;
  auto& nodalValues = l_nodalValues.view();

  sierra::nalu::nodal_vector_workview<p, double> l_nodalVecValues;
  auto& nodalVecValues = l_nodalVecValues.view();

  TensorPoly polys[3] = { TensorPoly(p), TensorPoly(p), TensorPoly(p) };
  for (int k = 0; k < p + 1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];
        nodalValues(k, j, i) = polys[0](locx, locy, locz);
        nodalVecValues(k, j, i, 0) = polys[0](locx, locy, locz);
        nodalVecValues(k, j, i, 1) = polys[1](locx, locy, locz);
        nodalVecValues(k, j, i, 2) = polys[2](locx, locy, locz);
      }
    }
  }


  const auto scsLocs = sierra::nalu::gauss_legendre_rule(p).first;

  sierra::nalu::nodal_scalar_workview<p, double> l_operator_scalar_interp(0);
  auto& operator_scalar_interp = l_operator_scalar_interp.view();

  sierra::nalu::nodal_vector_workview<p, double> l_operator_vector_interp(0);
  auto& operator_vector_interp = l_operator_vector_interp.view();

  ops.scs_xhat_interp(nodalValues, operator_scalar_interp);
  ops.scs_xhat_interp(nodalVecValues, operator_vector_interp);

  for (int k = 0; k < p +1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p; ++i) {
        double locx = scsLocs[i];
        ASSERT_NEAR(operator_scalar_interp(k,j,i), polys[0](locx,locy,locz), my_tol)<< "x";
        for (int d = 0; d < 3; ++d) {
          ASSERT_NEAR(operator_vector_interp(k,j,i,d), polys[d](locx,locy,locz), my_tol) << "x, " << d;
        }
      }
    }
  }

  ops.scs_yhat_interp(nodalValues, operator_scalar_interp);
  ops.scs_yhat_interp(nodalVecValues, operator_vector_interp);

  for (int k = 0; k < p +1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p; ++j) {
      double locy = scsLocs[j];
      for (int i = 0; i < p+1; ++i) {
        double locx = elem->nodeLocs1D[i];
        ASSERT_NEAR(operator_scalar_interp(k,j,i), polys[0](locx,locy,locz),my_tol) << "y";
        for (int d = 0; d < 3; ++d) {
          ASSERT_NEAR(operator_vector_interp(k,j,i,d), polys[d](locx,locy,locz),my_tol) << "y, " << d;
        }
      }
    }
  }

  ops.scs_zhat_interp(nodalValues, operator_scalar_interp);
  ops.scs_zhat_interp(nodalVecValues, operator_vector_interp);

  for (int k = 0; k < p; ++k) {
    double locz = scsLocs[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p+1; ++i) {
        double locx = elem->nodeLocs1D[i];
        ASSERT_NEAR(operator_scalar_interp(k,j,i), polys[0](locx,locy,locz),my_tol) << "z";
        for (int d = 0; d < 3; ++d) {
          ASSERT_NEAR(operator_vector_interp(k,j,i,d), polys[d](locx,locy,locz),my_tol) << "z, " << d;;
        }
      }
    }
  }
}
//--------------------------------------------------------------------------
template <int p> void scs_grad_hex()
{
  auto ops = sierra::nalu::CVFEMOperators<p, double>();
  auto elem = sierra::nalu::ElementDescription::create(3, p);

  sierra::nalu::nodal_scalar_workview<p, double> l_nodalValues(0);
  auto& nodalValues = l_nodalValues.view();

  sierra::nalu::nodal_vector_workview<p, double> l_nodalVecValues(0);
  auto& nodalVecValues = l_nodalVecValues.view();

  const auto nodeLocs1D = sierra::nalu::gauss_lobatto_legendre_rule(p).first;
  const auto scsLocs = sierra::nalu::gauss_legendre_rule(p).first;

  TensorPoly polys[3] = { TensorPoly(p), TensorPoly(p), TensorPoly(p) };
  for (int k = 0; k < p + 1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];
        nodalValues(k, j, i) = polys[0](locx, locy, locz);
        nodalVecValues(k, j, i, 0) = polys[0](locx, locy, locz);
        nodalVecValues(k, j, i, 1) = polys[1](locx, locy, locz);
        nodalVecValues(k, j, i, 2) = polys[2](locx, locy, locz);
      }
    }
  }

  sierra::nalu::nodal_vector_workview<p, double> l_op_grad(0);
  auto& op_grad =  l_op_grad.view();

  sierra::nalu::nodal_tensor_workview<p, double> l_op_gradv(0);
  auto& op_gradv = l_op_gradv.view();

  ops.scs_xhat_grad(nodalValues, op_grad);
  ops.scs_xhat_grad(nodalVecValues, op_gradv);

  for (int k = 0; k < p + 1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p; ++i) {
        double locx = scsLocs[i];
        ASSERT_NEAR( op_grad(k,j,i,0), polys[0].grad_x(locx,locy,locz), my_tol) << "x(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_grad(k,j,i,1), polys[0].grad_y(locx,locy,locz), my_tol) << "x(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_grad(k,j,i,2), polys[0].grad_z(locx,locy,locz), my_tol) << "x(k,j,i) = (" << k << ", " << j << ", " << i << ")";

        for (int d = 0; d < 3; ++ d) {
          ASSERT_NEAR( op_gradv(k,j,i,d,0), polys[d].grad_x(locx,locy,locz), my_tol) << "x(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
          ASSERT_NEAR( op_gradv(k,j,i,d,1), polys[d].grad_y(locx,locy,locz), my_tol) << "x(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
          ASSERT_NEAR( op_gradv(k,j,i,d,2), polys[d].grad_z(locx,locy,locz), my_tol) << "x(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
        }
      }
    }
  }

  ops.scs_yhat_grad(nodalValues, op_grad);
  ops.scs_yhat_grad(nodalVecValues, op_gradv);
  for (int k = 0; k < p + 1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p ; ++j) {
      double locy = scsLocs[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];
        ASSERT_NEAR( op_grad(k,j,i,0), polys[0].grad_x(locx,locy,locz), my_tol) << "y(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_grad(k,j,i,1), polys[0].grad_y(locx,locy,locz), my_tol) << "y(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_grad(k,j,i,2), polys[0].grad_z(locx,locy,locz), my_tol)<< "y(k,j,i) = (" << k << ", " << j << ", " << i << ")";

        for (int d = 0; d < 3; ++d) {
          ASSERT_NEAR( op_gradv(k,j,i,d, 0), polys[d].grad_x(locx,locy,locz), my_tol)<< "y(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
          ASSERT_NEAR( op_gradv(k,j,i,d, 1), polys[d].grad_y(locx,locy,locz), my_tol) << "y(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
          ASSERT_NEAR( op_gradv(k,j,i,d, 2), polys[d].grad_z(locx,locy,locz), my_tol) << "y(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
        }
      }
    }
  }
//
  ops.scs_zhat_grad(nodalValues, op_grad);
  ops.scs_zhat_grad(nodalVecValues, op_gradv);
  for (int k = 0; k < p; ++k) {
    double locz = scsLocs[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];;
        ASSERT_NEAR( op_grad(k,j,i,0), polys[0].grad_x(locx,locy,locz), my_tol) << "z(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_grad(k,j,i,1), polys[0].grad_y(locx,locy,locz), my_tol) << "z(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_grad(k,j,i,2), polys[0].grad_z(locx,locy,locz), my_tol) << "z(k,j,i) = (" << k << ", " << j << ", " << i << ")";

        for (int d = 0; d < 3; ++ d) {
          ASSERT_NEAR( op_gradv(k,j,i,d,0), polys[d].grad_x(locx,locy,locz), my_tol) << "z(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
          ASSERT_NEAR( op_gradv(k,j,i,d,1), polys[d].grad_y(locx,locy,locz), my_tol) << "z(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
          ASSERT_NEAR( op_gradv(k,j,i,d,2), polys[d].grad_z(locx,locy,locz), my_tol) << "z(d,k,j,i) = (" << d << ", " << k << ", " << j << ", " << i << ")";
        }
      }
    }
  }

}



//--------------------------------------------------------------------------
template <int p> void nodal_grad_hex()
{
  auto ops = sierra::nalu::CVFEMOperators<p, double>();
  auto elem = sierra::nalu::ElementDescription::create(3, p);

  sierra::nalu::nodal_scalar_workview<p, double> l_nodalValues(0);
  auto& nodalValues = l_nodalValues.view();

  const auto nodeLocs1D = sierra::nalu::gauss_lobatto_legendre_rule(p).first;

  TensorPoly poly(p);
  for (int k = 0; k < p + 1; ++k) {
    double locz =elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];
        nodalValues(k,j,i) = poly(locx,locy,locz);
      }
    }
  }

  sierra::nalu::nodal_vector_workview<p, double> l_op_nodal_grad(0);
  auto& op_nodal_grad =  l_op_nodal_grad.view();
  ops.nodal_grad(nodalValues, op_nodal_grad);
  for (int k = 0; k < p + 1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];
        ASSERT_NEAR( op_nodal_grad(k,j,i,0), poly.grad_x(locx,locy,locz), my_tol) << "(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_nodal_grad(k,j,i,1), poly.grad_y(locx,locy,locz), my_tol) << "(k,j,i) = (" << k << ", " << j << ", " << i << ")";
        ASSERT_NEAR( op_nodal_grad(k,j,i,2), poly.grad_z(locx,locy,locz), my_tol) << "(k,j,i) = (" << k << ", " << j << ", " << i << ")";
      }
    }
  }
}

////--------------------------------------------------------------------------
template <int p> void scv_integration_hex()
{
  auto ops = sierra::nalu::CVFEMOperators<p, double>();

  auto elem = sierra::nalu::ElementDescription::create(3, p);
  sierra::nalu::nodal_scalar_workview<p, double> l_nodalValues(0);
  auto& nodalValues = l_nodalValues.view();

  sierra::nalu::nodal_vector_workview<p, double> l_nodalVecValues(0);
  auto& nodalVecValues = l_nodalVecValues.view();

  const auto nodeLocs1D = sierra::nalu::gauss_lobatto_legendre_rule(p).first;
  const auto scsLocs = sierra::nalu::gauss_legendre_rule(p).first;

  TensorPoly polys[3] = { TensorPoly(p), TensorPoly(p), TensorPoly(p) };

  for (int k = 0; k < p + 1; ++k) {
    double locz = elem->nodeLocs1D[k];
    for (int j = 0; j < p + 1; ++j) {
      double locy = elem->nodeLocs1D[j];
      for (int i = 0; i < p + 1; ++i) {
        double locx = elem->nodeLocs1D[i];
        nodalValues(k, j, i) = polys[0](locx, locy, locz);
        nodalVecValues(k, j, i, 0) = polys[0](locx, locy, locz);
        nodalVecValues(k, j, i, 1) = polys[1](locx, locy, locz);
        nodalVecValues(k, j, i, 2) = polys[2](locx, locy, locz);
      }
    }
  }

  const auto scsEndLoc = sierra::nalu::pad_end_points(sierra::nalu::gauss_legendre_rule(p).first);

  sierra::nalu::nodal_scalar_workview<p, double> l_op_vol_integral(0);
  auto& op_vol_integral = l_op_vol_integral.view();

  sierra::nalu::nodal_vector_workview<p, double> l_op_volvec_integral(0);
  auto& op_volvec_integral = l_op_volvec_integral.view();

  ops.volume(nodalValues, op_vol_integral);
  ops.volume(nodalVecValues, op_volvec_integral);

  for (unsigned k = 0; k < p + 1; ++k) {
    double zl = scsEndLoc[k + 0];
    double zr = scsEndLoc[k + 1];
    for (unsigned j = 0; j < p + 1; ++j) {
      double yl = scsEndLoc[j + 0];
      double yr = scsEndLoc[j + 1];
      for (unsigned i = 0; i < p + 1; ++i) {
        double xl = scsEndLoc[i + 0];
        double xr = scsEndLoc[i + 1];
        ASSERT_NEAR(op_vol_integral(k,j,i), polys[0].xyz_integral(xl,xr,yl,yr,zl,zr), my_tol);
        ASSERT_NEAR(op_volvec_integral(k, j, i, 0), polys[0].xyz_integral(xl, xr, yl, yr, zl, zr), my_tol);
        ASSERT_NEAR(op_volvec_integral(k, j, i, 1), polys[1].xyz_integral(xl, xr, yl, yr, zl, zr), my_tol);
        ASSERT_NEAR(op_volvec_integral(k, j, i, 2), polys[2].xyz_integral(xl, xr, yl, yr, zl, zr), my_tol);
      }
    }
  }
}


}
TEST_POLY(HOOperators, scs_interp_hex, 5)
TEST_POLY(HOOperators, scs_grad_hex, 5)
TEST_POLY(HOOperators, nodal_grad_hex, 5)
TEST_POLY(HOOperators, scv_integration_hex, 5)
