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
#include <master_element/MasterElementHO.h>
#include <master_element/HexPCVFEM.h>

#include <element_promotion/QuadratureRule.h>
#include <element_promotion/LagrangeBasis.h>
#include <element_promotion/TensorProductQuadratureRule.h>

#include "UnitTestUtils.h"

namespace {

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


//-------------------------------------------------------------------------
double linear_scalar_value(int dim, double a, const double* b, const double* x)
{
  if (dim == 2u) {
    return (a + b[0] * x[0] + b[1] * x[1]);
  }
  return (a + b[0] * x[0] + b[1] * x[1] + b[2] * x[2]);
}
//-------------------------------------------------------------------------
struct LinearField
{
  LinearField(int in_dim, double in_a, const double* in_b) : dim(in_dim), a(in_a) {
    b[0] = in_b[0];
    b[1] = in_b[1];
    if (dim == 3) b[2] = in_b[2];
  }

  double operator()(const double* x) { return linear_scalar_value(dim, a, b, x); }

  const int dim;
  const double a;
  double b[3];
};

LinearField make_random_linear_field(int dim, std::mt19937& rng)
{

  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  double a = coeff(rng);
  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }
  return LinearField(dim, a, coeffs.data());
}

//--------------------------------------------------------------------------
void
check_interpolation_quad(int polyOrder, int numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and interpolate to a vector of random points
  // (between -1.05 and 1.05 to check edges)

  auto elemDesc = sierra::nalu::ElementDescription::create(2, polyOrder);
  auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::uniform_real_distribution<double> loc(-1.05, 1.05);

  int nodesPerElement = elemDesc->nodesPerElement;
  Kokkos::View<double**> intgLoc("intg", numIps, elemDesc->dimension);
  std::vector<double> coeffsX(elemDesc->polyOrder+1);
  std::vector<double> coeffsY(elemDesc->polyOrder+1);
  std::vector<double> nodalValues(nodesPerElement);
  std::vector<double> exactInterp(numIps);
  std::vector<double> approxInterp(numIps);

  for (int ip = 0; ip < numIps; ++ip) {
    intgLoc(ip, 0) = loc(rng);
    intgLoc(ip, 1) = loc(rng);
  }

  for (int k = 0; k < elemDesc->polyOrder+1; ++k) {
    coeffsX[k] = coeff(rng);
    coeffsY[k] = coeff(rng);
  }

  auto interpWeights = basis.eval_basis_weights(intgLoc);

  for (int i = 0; i < elemDesc->nodes1D; ++i) {
    for (int j = 0; j < elemDesc->nodes1D; ++j) {
      nodalValues[elemDesc->node_map(i, j)] =
            poly_val(coeffsX, elemDesc->nodeLocs1D[i])
          * poly_val(coeffsY, elemDesc->nodeLocs1D[j]);
    }
  }

  for (int ip = 0; ip < numIps; ++ip) {
    exactInterp[ip] =
          poly_val(coeffsX, intgLoc(ip, 0))
        * poly_val(coeffsY, intgLoc(ip, 1));
  }

  for (int ip = 0; ip < numIps; ++ip) {
    double val = 0.0;
    for (int nodeNumber = 0; nodeNumber < nodesPerElement; ++nodeNumber) {
      val += interpWeights(ip, nodeNumber) * nodalValues[nodeNumber];
    }
    approxInterp[ip] = val;
  }

  for (unsigned j = 0; j < exactInterp.size(); ++j) {
    EXPECT_NEAR(approxInterp[j], exactInterp[j], tol);
  }
}
//--------------------------------------------------------------------------
void
check_interpolation_hex(int polyOrder, int numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and interpolate to a vector of random points
  // (between -1.05 and 1.05 to check edges)
  auto elemDesc = sierra::nalu::ElementDescription::create(3, polyOrder);
  auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::uniform_real_distribution<double> loc(-1.05, 1.05);

  int nodesPerElement = elemDesc->nodesPerElement;
  Kokkos::View<double**> intgLoc("intg", numIps, elemDesc->dimension);
  std::vector<double> coeffsX(elemDesc->polyOrder+1);
  std::vector<double> coeffsY(elemDesc->polyOrder+1);
  std::vector<double> coeffsZ(elemDesc->polyOrder+1);
  std::vector<double> nodalValues(nodesPerElement);
  std::vector<double> exactInterp(numIps);
  std::vector<double> approxInterp(numIps);

  for (int ip = 0; ip < numIps; ++ip) {
    intgLoc(ip, 0) = loc(rng);
    intgLoc(ip, 1) = loc(rng);
    intgLoc(ip, 2) = loc(rng);
  }

  for (int k = 0; k < elemDesc->polyOrder+1; ++k) {
    coeffsX[k] = coeff(rng);
    coeffsY[k] = coeff(rng);
    coeffsZ[k] = coeff(rng);
  }

  auto interpWeights = basis.eval_basis_weights(intgLoc);

  for (int i = 0; i < elemDesc->nodes1D; ++i) {
    for (int j = 0; j < elemDesc->nodes1D; ++j) {
      for (int k = 0; k < elemDesc->nodes1D; ++k) {
        nodalValues[elemDesc->node_map(i, j, k)] =
            poly_val(coeffsX, elemDesc->nodeLocs1D[i])
            * poly_val(coeffsY, elemDesc->nodeLocs1D[j])
            * poly_val(coeffsZ, elemDesc->nodeLocs1D[k]);
      }
    }
  }

  for (int ip = 0; ip < numIps; ++ip) {
    exactInterp[ip] =
          poly_val(coeffsX, intgLoc(ip, 0))
        * poly_val(coeffsY, intgLoc(ip, 1))
        * poly_val(coeffsZ, intgLoc(ip, 2));
  }

  for (int ip = 0; ip < numIps; ++ip) {
    double val = 0.0;
    for (int nodeNumber = 0; nodeNumber < nodesPerElement; ++nodeNumber) {
      val += interpWeights(ip, nodeNumber) * nodalValues[nodeNumber];
    }
    approxInterp[ip] = val;
  }

  for (unsigned j = 0; j < exactInterp.size(); ++j) {
    EXPECT_NEAR(approxInterp[j], exactInterp[j], tol);
  }
}
//--------------------------------------------------------------------------
void
check_derivative_quad(int polyOrder, int numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and compute derivatives at a vector of random points
  // (between -1.05 and 1.05 to check edges)
  auto elemDesc = sierra::nalu::ElementDescription::create(2, polyOrder);
  auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0,1.0);
  std::uniform_real_distribution<double> loc(-1.05,1.05);

  Kokkos::View<double**> intgLoc("intg", numIps, elemDesc->dimension);
  std::vector<double> coeffsX(elemDesc->polyOrder+1);
  std::vector<double> coeffsY(elemDesc->polyOrder+1);
  std::vector<double> exactDeriv(numIps*elemDesc->dimension);
  std::vector<double> approxDeriv(numIps*elemDesc->dimension);

  for (int ip = 0; ip < numIps; ++ip) {
    intgLoc(ip,0) = loc(rng);
    intgLoc(ip,1) = loc(rng);
  }

  auto diffWeights = basis.eval_deriv_weights(intgLoc);

  for (int k = 0; k < elemDesc->polyOrder+1; ++k) {
    coeffsX[k] = coeff(rng);
    coeffsY[k] = coeff(rng);
  }

  // create a (-1,1) x (-1,1) element and fill it with polynomial values
  // expect exact values to floating-point precision
  std::vector<double> nodalValues(elemDesc->nodesPerElement);
  for (int i = 0; i < elemDesc->nodes1D; ++i) {
    for(int j = 0; j < elemDesc->nodes1D; ++j) {
      nodalValues[elemDesc->node_map(i,j)] =
          poly_val(coeffsX,elemDesc->nodeLocs1D[i]) * poly_val(coeffsY,elemDesc->nodeLocs1D[j]);
    }
  }

  for (int ip = 0; ip < numIps; ++ip) {
    int offset = ip*elemDesc->dimension;
    exactDeriv[offset + 0] =
          poly_der(coeffsX, intgLoc(ip, 0))
        * poly_val(coeffsY, intgLoc(ip, 1));
    exactDeriv[offset + 1] =
          poly_val(coeffsX, intgLoc(ip, 0))
        * poly_der(coeffsY, intgLoc(ip, 1));
  }

  for (int ip = 0; ip < numIps; ++ip) {
    double dndx = 0.0;
    double dndy = 0.0;
    for (int node = 0; node < elemDesc->nodesPerElement; ++node) {
      dndx += diffWeights(ip, node, 0) * nodalValues[node];
      dndy += diffWeights(ip, node, 1) * nodalValues[node];
    }
    approxDeriv[ip*elemDesc->dimension+0] = dndx;
    approxDeriv[ip*elemDesc->dimension+1] = dndy;
  }

  for (unsigned j = 0; j < exactDeriv.size(); ++j) {
    EXPECT_NEAR(approxDeriv[j], exactDeriv[j], tol);
  }
}
//--------------------------------------------------------------------------
void
check_derivative_hex(int polyOrder, int numIps, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and compute derivatives at a vector of random points
  // (between -1.05 and 1.05 to check edges)
  auto elemDesc = sierra::nalu::ElementDescription::create(3, polyOrder);
  auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0,1.0);
  std::uniform_real_distribution<double> loc(-1.05,1.05);

  std::vector<double> intgLoc(numIps*elemDesc->dimension);
  std::vector<double> coeffsX(elemDesc->polyOrder+1);
  std::vector<double> coeffsY(elemDesc->polyOrder+1);
  std::vector<double> coeffsZ(elemDesc->polyOrder+1);
  std::vector<double> exactDeriv(numIps*elemDesc->dimension);
  std::vector<double> approxDeriv(numIps*elemDesc->dimension);

  for (int ip = 0; ip < numIps; ++ip) {
    int offset = ip*elemDesc->dimension;
    intgLoc[offset+0] = loc(rng);
    intgLoc[offset+1] = loc(rng);
    intgLoc[offset+2] = loc(rng);
  }

  auto diffWeights = basis.eval_deriv_weights(intgLoc.data(), numIps);

  for (int k = 0; k < elemDesc->polyOrder+1; ++k) {
    coeffsX[k] = coeff(rng);
    coeffsY[k] = coeff(rng);
    coeffsZ[k] = coeff(rng);
  }

  std::vector<double> nodalValues(elemDesc->nodesPerElement);
  for (int i = 0; i < elemDesc->nodes1D; ++i) {
    for(int j = 0; j < elemDesc->nodes1D; ++j) {
      for(int k = 0; k < elemDesc->nodes1D; ++k) {
        nodalValues[elemDesc->node_map(i,j,k)] =
              poly_val(coeffsX,elemDesc->nodeLocs1D[i])
            * poly_val(coeffsY,elemDesc->nodeLocs1D[j])
            * poly_val(coeffsZ,elemDesc->nodeLocs1D[k]);
      }
    }
  }

  for (int ip = 0; ip < numIps; ++ip) {
    int offset = ip*elemDesc->dimension;
    exactDeriv[offset + 0] =
          poly_der(coeffsX, intgLoc[offset])
        * poly_val(coeffsY, intgLoc[offset + 1])
        * poly_val(coeffsZ, intgLoc[offset + 2]);
    exactDeriv[offset + 1] =
          poly_val(coeffsX, intgLoc[offset])
        * poly_der(coeffsY, intgLoc[offset + 1])
        * poly_val(coeffsZ, intgLoc[offset + 2]);
    exactDeriv[offset + 2] =
          poly_val(coeffsX, intgLoc[offset])
        * poly_val(coeffsY, intgLoc[offset + 1])
        * poly_der(coeffsZ, intgLoc[offset + 2]);
  }

  for (int ip = 0; ip < numIps; ++ip) {
    double dndx = 0.0;
    double dndy = 0.0;
    double dndz = 0.0;
    for (int node = 0; node < elemDesc->nodesPerElement; ++node) {
      dndx += diffWeights(ip, node, 0) * nodalValues[node];
      dndy += diffWeights(ip, node, 1) * nodalValues[node];
      dndz += diffWeights(ip, node, 2) * nodalValues[node];
    }
    approxDeriv[ip*elemDesc->dimension+0] = dndx;
    approxDeriv[ip*elemDesc->dimension+1] = dndy;
    approxDeriv[ip*elemDesc->dimension+2] = dndz;
  }

  for (unsigned j = 0; j < exactDeriv.size(); ++j) {
    EXPECT_NEAR(approxDeriv[j], exactDeriv[j], tol);
  }
}
//--------------------------------------------------------------------------
void
check_volume_quadrature_quad(int polyOrder, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  auto elemDesc = sierra::nalu::ElementDescription::create(2, polyOrder);
  auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(elemDesc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderQuad2DSCV(basis, quad);

  const auto* interpWeights = masterElement.shape_functions();
  const auto* ipWeights = masterElement.ip_weights();
  const auto* ipNodeMap = masterElement.ipNodeMap();
  const auto& scsEndLoc = quad.scs_end_loc();
  std::vector<double> approxInt(elemDesc->nodesPerElement);
  std::vector<double> coeffsX(elemDesc->polyOrder+1);
  std::vector<double> coeffsY(elemDesc->polyOrder+1);
  std::vector<double> exactInt(elemDesc->nodesPerElement);
  std::vector<double> nodalValues(elemDesc->nodesPerElement);

  for (int k = 0; k < elemDesc->polyOrder+1; ++k) {
    coeffsX[k] = coeff(rng);
    coeffsY[k] = coeff(rng);
  }

  for (int i = 0; i < elemDesc->nodes1D; ++i) {
    for (int j = 0; j < elemDesc->nodes1D; ++j) {
      nodalValues[elemDesc->node_map(i, j)] =
            poly_val(coeffsX, elemDesc->nodeLocs1D[i])
          * poly_val(coeffsY, elemDesc->nodeLocs1D[j]);

      exactInt[elemDesc->node_map(i, j)] =
            poly_int(coeffsX, scsEndLoc[i], scsEndLoc[i + 1])
          * poly_int(coeffsY, scsEndLoc[j], scsEndLoc[j + 1]);
    }
  }

  approxInt.assign(approxInt.size(), 0.0);
  for (int ip = 0; ip < masterElement.num_integration_points(); ++ip) {
    double interpValue = 0.0;
    for (int nodeNumber = 0; nodeNumber < elemDesc->nodesPerElement; ++nodeNumber) {
      interpValue += interpWeights[ip*elemDesc->nodesPerElement+nodeNumber] * nodalValues[nodeNumber];
    }
    approxInt[ipNodeMap[ip]] += ipWeights[ip] * interpValue; //ipweights -> ws_scv_volume if not square
  }

  for (unsigned j = 0; j < exactInt.size(); ++j) {
    EXPECT_NEAR(approxInt[j], exactInt[j], tol);
  }
}
  //--------------------------------------------------------------------------
void
check_volume_quadrature_hex(int polyOrder, double tol)
{
  // create a (-1,1) x (-1,1) element filled with polynomial values
  // and integrate the polynomial over the dual nodal volumes

  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<double> coeff(-10.0, 10.0);

  auto elemDesc = sierra::nalu::ElementDescription::create(3, polyOrder);
  auto basis = sierra::nalu::LagrangeBasis(elemDesc->inverseNodeMap, elemDesc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(elemDesc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderHexSCV(basis, quad);

  const auto* interpWeights = masterElement.shape_functions();
  const auto* ipWeights = masterElement.ip_weights();
  const auto* ipNodeMap = masterElement.ipNodeMap();
  const auto& scsEndLoc = quad.scs_end_loc();
  std::vector<double> approxInt(elemDesc->nodesPerElement);
  std::vector<double> coeffsX(elemDesc->polyOrder+1);
  std::vector<double> coeffsY(elemDesc->polyOrder+1);
  std::vector<double> coeffsZ(elemDesc->polyOrder+1);
  std::vector<double> exactInt(elemDesc->nodesPerElement);
  std::vector<double> nodalValues(elemDesc->nodesPerElement);

  for (int k = 0; k < elemDesc->polyOrder+1; ++k) {
    coeffsX[k] = coeff(rng);
    coeffsY[k] = coeff(rng);
    coeffsZ[k] = coeff(rng);
  }

  for (int i = 0; i < elemDesc->nodes1D; ++i) {
    for (int j = 0; j < elemDesc->nodes1D; ++j) {
      for (int k = 0; k < elemDesc->nodes1D; ++k) {
        nodalValues[elemDesc->node_map(i, j, k)] =
            poly_val(coeffsX, elemDesc->nodeLocs1D[i])
            * poly_val(coeffsY, elemDesc->nodeLocs1D[j])
            * poly_val(coeffsZ, elemDesc->nodeLocs1D[k]);

        exactInt[elemDesc->node_map(i, j, k)] =
              poly_int(coeffsX, scsEndLoc[i], scsEndLoc[i + 1])
            * poly_int(coeffsY, scsEndLoc[j], scsEndLoc[j + 1])
            * poly_int(coeffsZ, scsEndLoc[k], scsEndLoc[k + 1]);
      }
    }
  }

  approxInt.assign(approxInt.size(), 0.0);
  for (int ip = 0; ip < masterElement.num_integration_points(); ++ip) {
    double interpValue = 0.0;
    for (int nodeNumber = 0; nodeNumber < elemDesc->nodesPerElement; ++nodeNumber) {
      interpValue += interpWeights[ip*elemDesc->nodesPerElement+nodeNumber] * nodalValues[nodeNumber];
    }
    approxInt[ipNodeMap[ip]] +=  ipWeights[ip]*interpValue;
  }

  for (unsigned j = 0; j < exactInt.size(); ++j) {
    EXPECT_NEAR(approxInt[j], exactInt[j], tol);
  }
}

void check_is_in_element_hex(int poly_order, double tol)
{
  auto desc = sierra::nalu::ElementDescription::create(3,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderHexSCS(basis, quad);
  constexpr int dim = 3;

  std::mt19937 rng;
  rng.seed(0);

  // randomly select a point within (boxmin, boxmax)^3 \subset reference element domain
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> random_pt(dim);
  for (int j = 0; j < dim; ++j) {
    random_pt[j] = coeff(rng);
  }

  // is in element uses a different stride for the coordinate data
  // compared to the gradient computation
  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  // Hex8/Quad4's is_in_element and interpolatePoint routines use a different,
  // but self-consistent reference element compared to the core shape functions
  // and derivatives
  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      for (int k = 0; k < desc->nodes1D; ++k) {
        int index = desc->node_map(i,j,k);
        ws_coords[0 * desc->nodesPerElement + index] = desc->nodeLocs1D[i];
        ws_coords[1 * desc->nodesPerElement + index] = desc->nodeLocs1D[j];
        ws_coords[2 * desc->nodesPerElement + index] = desc->nodeLocs1D[k];
      }
    }
  }

  std::array<double, dim> mePt;
  auto dist = masterElement.isInElement(ws_coords.data(), random_pt.data(), mePt.data());
  EXPECT_LT(dist, 1.0 + tol);
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(random_pt[d], mePt[d], tol);
  }
}

void check_is_not_in_element_hex(int poly_order, double tol)
{
  constexpr int dim = 3;
  auto desc = sierra::nalu::ElementDescription::create(dim,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderHexSCS(basis, quad);


  std::array<double, dim> exterior_pt = {{ 100., 100., 100.}};

  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      for (int k = 0; k < desc->nodes1D; ++k) {
        int index = desc->node_map(i,j,k);
        ws_coords[0 * desc->nodesPerElement + index] = desc->nodeLocs1D[i];
        ws_coords[1 * desc->nodesPerElement + index] = desc->nodeLocs1D[j];
        ws_coords[2 * desc->nodesPerElement + index] = desc->nodeLocs1D[k];
      }
    }
  }

  std::array<double, dim> mePt;
  double dist = masterElement.isInElement(ws_coords.data(), exterior_pt.data(), mePt.data());
  EXPECT_GT(dist, 1 + tol);
}

void check_point_interpolation_hex(int poly_order, double tol)
{
  auto desc = sierra::nalu::ElementDescription::create(3,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderHexSCS(basis, quad);
  constexpr int dim = 3;

  std::mt19937 rng;
  rng.seed(0);

  // randomly select a point within (boxmin, boxmax)^3 \subset reference element domain
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> random_pt(dim);
  for (int j = 0; j < dim; ++j) {
    random_pt[j] = coeff(rng);
  }

  double const_value = coeff(rng);
  std::array<double, 3> coeffs;
  for (int d = 0; d < dim; ++d) {
    coeffs[d] = coeff(rng);
  }

  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  const double delta = 0.25;
  std::uniform_real_distribution<double> coord_perturb(-delta/2, delta/2);

  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      for (int k = 0; k < desc->nodes1D; ++k) {
        std::array<double, 3> perturbed_coords = {{
            desc->nodeLocs1D[i] + coord_perturb(rng),
            desc->nodeLocs1D[j] + coord_perturb(rng),
            desc->nodeLocs1D[k] + coord_perturb(rng)
        }};

        int index = desc->node_map(i,j,k);
        ws_field[index] = linear_scalar_value(dim, const_value , coeffs.data(), perturbed_coords.data());

        ws_coords[0 * desc->nodesPerElement + index] = perturbed_coords[0];
        ws_coords[1 * desc->nodesPerElement + index] = perturbed_coords[1];
        ws_coords[2 * desc->nodesPerElement + index] = perturbed_coords[2];
      }
    }
  }

  std::array<double, dim> mePt;
  double dist = masterElement.isInElement(ws_coords.data(), random_pt.data(), mePt.data());
  EXPECT_LT(dist, 1.0+tol);

  double meInterp = 0.0;
  masterElement.interpolatePoint(1, mePt.data(), ws_field.data(), &meInterp);
  double exactVal = linear_scalar_value(dim, const_value , coeffs.data(), random_pt.data());
  EXPECT_NEAR(meInterp, exactVal, tol);
}

void check_is_in_element_quad(int poly_order, double tol)
{
  constexpr int dim = 2;
  auto desc = sierra::nalu::ElementDescription::create(dim,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderQuad2DSCS(basis, quad);

  std::mt19937 rng;
  rng.seed(0);

  // randomly select a point within (boxmin, boxmax)^3 \subset reference element domain
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> random_pt(dim);
  for (int j = 0; j < dim; ++j) {
    random_pt[j] = coeff(rng);
  }

  // is in element uses a different stride for the coordinate data
  // compared to the gradient computation
  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  // Hex8/Quad4's is_in_element and interpolatePoint routines use a different,
  // but self-consistent reference element compared to the core shape functions
  // and derivatives
  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      int index = desc->node_map(i,j);
      ws_coords[0 * desc->nodesPerElement + index] = desc->nodeLocs1D[i];
      ws_coords[1 * desc->nodesPerElement + index] = desc->nodeLocs1D[j];
    }
  }

  std::array<double, dim> mePt;
  auto dist = masterElement.isInElement(ws_coords.data(), random_pt.data(), mePt.data());
  EXPECT_LT(dist, 1.0 + tol);
  for (int d = 0; d < dim; ++d) {
    EXPECT_NEAR(random_pt[d], mePt[d], tol);
  }
}

void check_is_not_in_element_quad(int poly_order, double tol)
{
  constexpr int dim = 2;
  auto desc = sierra::nalu::ElementDescription::create(dim,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderQuad2DSCS(basis, quad);


  std::array<double, dim> exterior_pt = {{ 100., 100.}};

  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      int index = desc->node_map(i,j);
      ws_coords[0 * desc->nodesPerElement + index] = desc->nodeLocs1D[i];
      ws_coords[1 * desc->nodesPerElement + index] = desc->nodeLocs1D[j];
    }
  }

  std::array<double, dim> mePt;
  auto dist = masterElement.isInElement(ws_coords.data(), exterior_pt.data(), mePt.data());
  EXPECT_GT(dist, 1 + tol);
}

void check_point_interpolation_quad(int poly_order, double tol)
{
  constexpr int dim = 2;
  auto desc = sierra::nalu::ElementDescription::create(dim,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderQuad2DSCS(basis, quad);

  std::mt19937 rng;
  rng.seed(0);

  // randomly select a point within (boxmin, boxmax)^3 \subset reference element domain
  const double boxmin = 0.125;
  const double boxmax = 0.25;
  std::uniform_real_distribution<double> coeff(boxmin, boxmax);
  std::vector<double> random_pt(dim);
  for (int j = 0; j < dim; ++j) {
    random_pt[j] = coeff(rng);
  }

  double const_value = coeff(rng);
  std::array<double, dim> coeffs;
  for (int d = 0; d < dim; ++d) {
    coeffs[d] = coeff(rng);
  }

  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  const double delta = 0.25;
  std::uniform_real_distribution<double> coord_perturb(-delta/2, delta/2);

  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      std::array<double, 2> perturbed_coords = {{
          desc->nodeLocs1D[i] + coord_perturb(rng),
          desc->nodeLocs1D[j] + coord_perturb(rng)
      }};

      int index = desc->node_map(i,j);
      ws_field[index] = linear_scalar_value(dim, const_value , coeffs.data(), perturbed_coords.data());

      ws_coords[0 * desc->nodesPerElement + index] = perturbed_coords[0];
      ws_coords[1 * desc->nodesPerElement + index] = perturbed_coords[1];
    }
  }

  std::array<double, dim> mePt;
  double dist = masterElement.isInElement(ws_coords.data(), random_pt.data(), mePt.data());
  EXPECT_LT(dist, 1.0+tol);

  double meInterp = 0.0;
  masterElement.interpolatePoint(1, mePt.data(), ws_field.data(), &meInterp);
  double exactVal = linear_scalar_value(dim, const_value , coeffs.data(), random_pt.data());
  EXPECT_NEAR(meInterp, exactVal, tol);
}

void check_scv_grad_op_hex(int poly_order, double tol)
{
  auto desc = sierra::nalu::ElementDescription::create(3,  poly_order);
  auto basis = sierra::nalu::LagrangeBasis(desc->inverseNodeMap, desc->nodeLocs1D);
  auto quad = sierra::nalu::TensorProductQuadratureRule(desc->polyOrder);
  auto masterElement = sierra::nalu::HigherOrderHexSCS(basis, quad);
  constexpr int dim = 3;

  std::vector<double> ws_field(desc->nodesPerElement);
  std::vector<double> ws_coords(desc->nodesPerElement * dim);

  std::mt19937 rng;
  rng.seed(0);
  auto linField = make_random_linear_field(dim, rng);

  const double delta = 0.25;
  std::uniform_real_distribution<double> coord_perturb(-delta/2, delta/2);


  for (int i = 0; i < desc->nodes1D; ++i) {
    for (int j = 0; j < desc->nodes1D; ++j) {
      for (int k = 0; k < desc->nodes1D; ++k) {
        int index = desc->node_map(i,j,k);

        std::array<double, 3> perturbed_coords = {{
            desc->nodeLocs1D[i] + coord_perturb(rng),
            desc->nodeLocs1D[j] + coord_perturb(rng),
            desc->nodeLocs1D[k] + coord_perturb(rng)
        }};

        ws_coords[index * dim + 0] = perturbed_coords[0];
        ws_coords[index * dim + 1] = perturbed_coords[1];
        ws_coords[index * dim + 2] = perturbed_coords[2];
        ws_field[index] = linField(&ws_coords[index*dim]);
      }
    }
  }

  const int derivSize = masterElement.num_integration_points()* desc->nodesPerElement * dim;
  std::vector<double> ws_gradop(derivSize);
  std::vector<double> ws_deriv(derivSize);
  std::vector<double> ws_detj(masterElement.num_integration_points());
  double error = 0;
  masterElement.grad_op(1,ws_coords.data(), ws_gradop.data(), ws_deriv.data(), ws_detj.data(), &error);

  Kokkos::View<double***> dndx_scv(ws_gradop.data(), masterElement.num_integration_points(), masterElement.nodesPerElement_, dim);

  for (int ip = 0; ip < masterElement.num_integration_points(); ++ip) {
    double dqdxIp[3] = {0,0,0};
    for (int n = 0; n < masterElement.nodesPerElement_; ++n) {
      for (int d = 0; d < dim; ++d) {
        dqdxIp[d] += dndx_scv(ip, n, d) * ws_field[n];
      }
    }

    for (int d = 0; d < dim; ++d) {
      EXPECT_NEAR(dqdxIp[d], linField.b[d], tol);
    }
  }
}


}//namespace

constexpr int MAX_POLY_ORDER = 5;
#define TEST_IPS(x, y, z) \
    TEST(HOMasterElements, x) \
{ \
    for (int p = 1; p < MAX_POLY_ORDER + 1; ++p) { \
      x(p, y, z); \
    } \
}

#define TEST_POLY_SINGLE(x, y) \
    TEST(HOMasterElements, x) \
{ \
    for (int p = 1; p < MAX_POLY_ORDER + 1; ++p) { \
      x(p, y); \
    } \
}

TEST_IPS(check_interpolation_quad, 10, 1.0e-10)
TEST_IPS(check_interpolation_hex, 10, 1.0e-10)
TEST_IPS(check_derivative_quad, 10, 1.0e-10)
TEST_IPS(check_derivative_hex, 10, 1.0e-10)
TEST_POLY_SINGLE(check_volume_quadrature_quad, 1.0e-10)
TEST_POLY_SINGLE(check_volume_quadrature_hex, 1.0e-10)
TEST_POLY_SINGLE(check_is_in_element_quad, 1.0e-10)
TEST_POLY_SINGLE(check_is_in_element_hex, 1.0e-10)
TEST_POLY_SINGLE(check_is_not_in_element_quad, 1.0e-10)
TEST_POLY_SINGLE(check_is_not_in_element_hex, 1.0e-10)
TEST_POLY_SINGLE(check_point_interpolation_quad, 1.0e-8)
TEST_POLY_SINGLE(check_point_interpolation_hex, 1.0e-8)
TEST_POLY_SINGLE(check_scv_grad_op_hex, 1.0e-8)
