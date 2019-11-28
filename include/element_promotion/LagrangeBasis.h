// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LagrangeBasis_h
#define LagrangeBasis_h

#include <Kokkos_Core.hpp>

#include <vector>

namespace sierra{
namespace nalu{

class Lagrange1D
{
public:
  Lagrange1D(const double* nodeLocs, int order);
  Lagrange1D(Kokkos::View<double*> nodeLocs);

  KOKKOS_FUNCTION
  ~Lagrange1D();

  double interpolation_weight(double x, unsigned nodeNumber) const;
  double derivative_weight(double x, unsigned nodeNumber) const;
private:
  void set_lagrange_weights();
  Kokkos::View<double*> nodeLocs_;
  Kokkos::View<double*> barycentricWeights_;
};

class LagrangeBasis
{
public:
  LagrangeBasis(const std::vector<std::vector<int>>& indicesMap, const std::vector<double>& nodeLocs);

  KOKKOS_FUNCTION
  ~LagrangeBasis();

  Kokkos::View<double**> eval_basis_weights(const double* intgLoc, int nInt) const;
  Kokkos::View<double**> eval_basis_weights(const Kokkos::View<double**>& intgLoc) const;
  Kokkos::View<double***> eval_deriv_weights(const double* intgLoc, int nInt) const;
  Kokkos::View<double***> eval_deriv_weights(const Kokkos::View<double**>& intgLoc) const;


  const Kokkos::View<double*>& point_interpolation_weights(const double* isoParCoords);
  const Kokkos::View<double**>& point_derivative_weights(const double* isoParCoords);

  KOKKOS_INLINE_FUNCTION
  unsigned order() const { return polyOrder_; }
  KOKKOS_INLINE_FUNCTION
  unsigned num_nodes() const { return numNodes_; }

private:
  void fill_interpolation_weights(const double* isoParCoord, double* weights) const;
  void fill_derivative_weights(const double* isoParCoord, double* weights) const;

  const unsigned polyOrder_;
  const Lagrange1D basis1D_;
  const unsigned numNodes1D_;
  const unsigned dim_;
  const unsigned numNodes_;

  Kokkos::View<int**> indicesMap_;
  Kokkos::View<double*> interpWeightsAtPoint_;
  Kokkos::View<double**> derivWeightsAtPoint_;
};


} // namespace nalu
} // namespace Sierra

#endif
