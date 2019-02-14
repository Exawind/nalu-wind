/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/LagrangeBasis.h>
#include <stk_util/util/ReportHandler.hpp>
#include <element_promotion/QuadratureRule.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// Lagrange1D - Provides the set of weights for interpolating and taking
// derivatives for a 1-dimensional element ordered from left-to-right
//===========================================================================
Lagrange1D::Lagrange1D(int poly_order)
{
  nodeLocs_ = gauss_lobatto_legendre_rule(poly_order+1).first;
  set_lagrange_weights();
}
//--------------------------------------------------------------------------
Lagrange1D::Lagrange1D(const double* nodeLocs, int order)
{
  nodeLocs_.resize(order+1);
  for (int j = 0; j < order + 1; ++j) {
    nodeLocs_[j] = nodeLocs[j];
  }
  set_lagrange_weights();
}
//--------------------------------------------------------------------------
Lagrange1D::Lagrange1D(std::vector<double> nodeLocs) : nodeLocs_(std::move(nodeLocs))
{
  set_lagrange_weights();
}
//--------------------------------------------------------------------------
Lagrange1D::~Lagrange1D() = default;
//--------------------------------------------------------------------------
void
Lagrange1D::set_lagrange_weights()
{
  const auto numNodes = nodeLocs_.size();
  lagrangeWeights_.assign(numNodes,1.0);
  for (unsigned i = 0; i < numNodes; ++i) {
    for (unsigned j = 0; j < numNodes; ++j) {
      if ( i != j ) {
        lagrangeWeights_[i] *= (nodeLocs_[i]-nodeLocs_[j]);
      }
    }
    lagrangeWeights_[i] = 1.0 / lagrangeWeights_[i];
  }
}
//--------------------------------------------------------------------------
double
Lagrange1D::interpolation_weight(double x, unsigned nodeNumber) const
{
  double numerator = 1.0;
  for (unsigned j = 0; j < nodeLocs_.size(); ++j) {
    if (j != nodeNumber) {
      numerator *= (x - nodeLocs_[j]);
    }
  }
  return (numerator * lagrangeWeights_[nodeNumber]);
}
//--------------------------------------------------------------------------
double
Lagrange1D::derivative_weight(double x, unsigned nodeNumber) const
{
  double outer = 0.0;
  for (unsigned j = 0; j < nodeLocs_.size(); ++j) {
    if (j != nodeNumber) {
      double inner = 1.0;
      for (unsigned i = 0; i < nodeLocs_.size(); ++i) {
        if (i != j && i != nodeNumber) {
          inner *= (x - nodeLocs_[i]);
        }
      }
      outer += inner;
    }
  }
  return (outer * lagrangeWeights_[nodeNumber]);
}
//--------------------------------------------------------------------------
LagrangeBasis::LagrangeBasis(
  const std::vector<std::vector<int>>& indicesMap,
  const std::vector<double>& nodeLocs)
  :  indicesMap_(indicesMap),
     basis1D_(nodeLocs),
     numNodes1D_(nodeLocs.size()),
     polyOrder_(numNodes1D_-1),
     dim_(indicesMap[0].size())
{
  for (auto& indices : indicesMap) {
    ThrowRequire(indices.size() == dim_);
  }

  numNodes_ = std::pow(numNodes1D_, dim_);

  interpWeightsAtPoint_.resize(numNodes_);
  derivWeightsAtPoint_.resize(numNodes_ * dim_);
}
//--------------------------------------------------------------------------
LagrangeBasis::~LagrangeBasis() = default;
//--------------------------------------------------------------------------
void LagrangeBasis::interpolation_weights(const double* isoParCoords, double* weights) const
{
  for (unsigned nodeNumber = 0; nodeNumber < numNodes_; ++nodeNumber) {
    const int* idx = indicesMap_[nodeNumber].data();
    weights[nodeNumber] = tensor_lagrange_interpolant(dim_, isoParCoords, idx);
  }
}
//--------------------------------------------------------------------------
void LagrangeBasis::derivative_weights(const double* isoParCoords, double* weights) const
{
  int derivIndex = 0;
  for (unsigned nodeNumber = 0; nodeNumber < numNodes_; ++nodeNumber) {
    const int* idx = indicesMap_[nodeNumber].data();
    for (unsigned d = 0; d < dim_; ++d) {
      weights[derivIndex] = tensor_lagrange_derivative(dim_, isoParCoords, idx, d);
      ++derivIndex;
    }
  }
}
//--------------------------------------------------------------------------
const std::vector<double>& LagrangeBasis::point_interpolation_weights(const double* isoParCoords)
{
  interpolation_weights(isoParCoords, interpWeightsAtPoint_.data());
  return interpWeightsAtPoint_;
}
//--------------------------------------------------------------------------
const std::vector<double>& LagrangeBasis::point_derivative_weights(const double* isoParCoords)
{
  derivative_weights(isoParCoords, derivWeightsAtPoint_.data());
  return derivWeightsAtPoint_;
}
//--------------------------------------------------------------------------
std::vector<double>
LagrangeBasis::eval_basis_weights(const std::vector<double>& intgLoc) const
{
  auto numIps = intgLoc.size() / dim_;
  ThrowAssert(numIps * dim_ == intgLoc.size());

  std::vector<double> interpolationWeights(numIps*numNodes_);
  for (unsigned ip = 0; ip < numIps; ++ip) {
    interpolation_weights(&intgLoc[ip*dim_], &interpolationWeights[ip*numNodes_]);
  }
  return interpolationWeights;
}
//--------------------------------------------------------------------------
std::vector<double>
LagrangeBasis::eval_deriv_weights(const std::vector<double>& intgLoc) const
{
  auto numIps = intgLoc.size()/dim_;
  auto numNodes = std::pow(numNodes1D_,dim_);
  std::vector<double> derivWeights(numIps * numNodes * dim_);

  for (unsigned ip = 0; ip < numIps; ++ip) {
    derivative_weights(&intgLoc[ip*dim_], &derivWeights[ip*dim_*numNodes_]);
  }
  return derivWeights;
}
//--------------------------------------------------------------------------
double
LagrangeBasis::tensor_lagrange_interpolant(
  unsigned dimension,
  const double* x,
  const int* node_ordinals) const
{
  double interpolant_weight = 1.0;
  for (unsigned j = 0; j < dimension; ++j) {
    interpolant_weight *= basis1D_.interpolation_weight(x[j], node_ordinals[j]);
  }
  return interpolant_weight;
}
//--------------------------------------------------------------------------
double
LagrangeBasis::tensor_lagrange_derivative(
  unsigned dimension,
  const double* x,
  const int* node_ordinals,
  unsigned derivativeDirection) const
{
  double derivativeWeight = 1.0;
  for (unsigned j = 0; j < dimension; ++j) {
    if (j == derivativeDirection) {
      derivativeWeight *= basis1D_.derivative_weight(x[j], node_ordinals[j]);
    }
    else {
      derivativeWeight *= basis1D_.interpolation_weight(x[j], node_ordinals[j]);
    }
  }
  return derivativeWeight;
}

}  // namespace naluUnit
} // namespace sierra
