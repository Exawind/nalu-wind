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


Kokkos::View<double*> barycentric_weights(const Kokkos::View<double*>& nodelocs)
{
  const auto numNodes = nodelocs.extent(0);
  auto barycentricWeights = Kokkos::View<double*>("barycentric_weights", numNodes);
  for (unsigned k = 0; k < numNodes; ++k) barycentricWeights[k] = 1.0;

  for (unsigned i = 0; i < numNodes; ++i) {
    for (unsigned j = 0; j < numNodes; ++j) {
      if ( i != j ) {
        barycentricWeights[i] *= (nodelocs[i] - nodelocs[j]);
      }
    }
    barycentricWeights[i] = 1.0 / barycentricWeights[i];
  }
  return barycentricWeights;
}

Lagrange1D::Lagrange1D(const double* nodeLocs, int p)
: nodeLocs_("node_locs", p + 1)
{
  for (int k = 0; k < p + 1; ++k) {
    nodeLocs_[k] = nodeLocs[k];
  }
  barycentricWeights_ = barycentric_weights(nodeLocs_);
}

Lagrange1D::Lagrange1D(Kokkos::View<double*> nodeLocs)
: nodeLocs_(nodeLocs), barycentricWeights_(barycentric_weights(nodeLocs)) {}

Lagrange1D::~Lagrange1D() = default;

double
Lagrange1D::interpolation_weight(double x, unsigned nodeNumber) const
{
  double numerator = 1.0;
  for (unsigned j = 0; j < nodeLocs_.size(); ++j) {
    if (j != nodeNumber) {
      numerator *= (x - nodeLocs_[j]);
    }
  }
  return (numerator * barycentricWeights_[nodeNumber]);
}

double Lagrange1D::derivative_weight(double x, unsigned nodeNumber) const
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
  return (outer * barycentricWeights_[nodeNumber]);
}

LagrangeBasis::LagrangeBasis(
  const std::vector<std::vector<int>>& indicesMap,
  const std::vector<double>& nodeLocs)
  :  polyOrder_(nodeLocs.size() - 1),
     basis1D_(nodeLocs.data(), polyOrder_),
     numNodes1D_(nodeLocs.size()),
     dim_(indicesMap[0].size()),
     numNodes_(std::pow(nodeLocs.size(), indicesMap[0].size())),
     indicesMap_("index_map", indicesMap.size(), indicesMap[0].size()),
     interpWeightsAtPoint_("interp_weights_at_point", numNodes_),
     derivWeightsAtPoint_("deriv_weights_at_point", numNodes_, dim_)
{
  for (unsigned k = 0; k < indicesMap.size(); ++k) {
    ThrowRequire(indicesMap[k].size() == dim_);
    for (unsigned d = 0; d < dim_; ++d) {
      indicesMap_(k, d) = indicesMap.at(k).at(d);
    }
  }
}

LagrangeBasis::~LagrangeBasis() = default;

void LagrangeBasis::fill_interpolation_weights(const double* isoParCoords, double* weights) const
{
  for (unsigned k = 0; k < numNodes_; ++k) {
    double interpolant_weight = 1.0;
    for (unsigned d = 0; d < dim_; ++d) {
      interpolant_weight *= basis1D_.interpolation_weight(isoParCoords[d], indicesMap_(k, d));
    }
    weights[k] = interpolant_weight;
  }
}

void LagrangeBasis::fill_derivative_weights(const double* isoParCoords, double* weights) const
{
  for (unsigned k = 0; k < numNodes_; ++k) {
    for (unsigned dj = 0; dj < dim_; ++dj) {
      double derivative_weight = 1.0;
      for (unsigned di = 0; di < dim_; ++di) {
        const double x = isoParCoords[di];
        const int mapped_k = indicesMap_(k, di);
        derivative_weight *= (di == dj) ?
            basis1D_.derivative_weight(x, mapped_k) : basis1D_.interpolation_weight(x, mapped_k);
      }
      weights[k * dim_ + dj] = derivative_weight;
    }
  }
}

const Kokkos::View<double*>& LagrangeBasis::point_interpolation_weights(const double* isoParCoords)
{
  fill_interpolation_weights(isoParCoords, interpWeightsAtPoint_.data());
  return interpWeightsAtPoint_;
}

const Kokkos::View<double**>& LagrangeBasis::point_derivative_weights(const double* isoParCoords)
{
  fill_derivative_weights(isoParCoords, derivWeightsAtPoint_.data());
  return derivWeightsAtPoint_;
}

Kokkos::View<double**> LagrangeBasis::eval_basis_weights(const double* intgLoc, int nInt) const
{
  ThrowAssert(numNodes_ > 0);
  ThrowAssert(nInt > 0);
  Kokkos::View<double**> interpWeights("interp_weights", nInt, numNodes_);
  for (int ip = 0; ip < nInt; ++ip) {
    fill_interpolation_weights(&intgLoc[ip*dim_], &interpWeights(ip, 0));
  }
  return interpWeights;
}

Kokkos::View<double**> LagrangeBasis::eval_basis_weights(const Kokkos::View<double**>& intgLoc) const
{
  return eval_basis_weights(intgLoc.data(), intgLoc.extent_int(0));
}

Kokkos::View<double***> LagrangeBasis::eval_deriv_weights(const double* intgLoc, int nInt) const
{
  Kokkos::View<double***> derivWeights("derivative_weights", nInt, numNodes_, dim_);
  for (int ip = 0; ip < nInt; ++ip) {
    fill_derivative_weights(&intgLoc[ip*dim_], &derivWeights(ip, 0, 0));
  }
  return derivWeights;
}

Kokkos::View<double***> LagrangeBasis::eval_deriv_weights(const Kokkos::View<double**>& intgLoc) const
{
  return eval_deriv_weights(intgLoc.data(), intgLoc.extent_int(0));
}


}  // namespace naluUnit
} // namespace sierra
