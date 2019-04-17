/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
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
  ~LagrangeBasis();
  Kokkos::View<double**> eval_basis_weights(const double* intgLoc, int nInt) const;
  Kokkos::View<double**> eval_basis_weights(const Kokkos::View<double**>& intgLoc) const;
  Kokkos::View<double***> eval_deriv_weights(const double* intgLoc, int nInt) const;
  Kokkos::View<double***> eval_deriv_weights(const Kokkos::View<double**>& intgLoc) const;


  const Kokkos::View<double*>& point_interpolation_weights(const double* isoParCoords);
  const Kokkos::View<double**>& point_derivative_weights(const double* isoParCoords);

  unsigned order() const { return polyOrder_; }
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
