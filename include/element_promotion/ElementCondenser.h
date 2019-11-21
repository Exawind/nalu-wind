// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ElementCondenser_h
#define ElementCondenser_h

#include <Teuchos_BLAS.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseVector.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_SerialDenseSolver.hpp>


namespace sierra {
namespace nalu {

  struct ElementDescription;

  class ElementCondenser
  {
  public:
    ElementCondenser(const ElementDescription& elem);

    void condense(
      double* lhs,
      const double* rhs,
      double* r_lhs,
      double* r_rhs
    );

    void compute_interior_update(
      double* lhs,
      const double* boundary_values,
      const double* rhs,
      double* interior_values
    );

    int num_boundary_nodes() { return nb_; }
    int num_internal_nodes() { return ni_; }
    int nodes_per_element()  { return ne_; }

  private:
    void chunk(const double* lhs, const double* rhs, double* b_lhs, double* b_rhs);
    void chunk_lower(const double* lhs, const double* rhs);

    Teuchos::BLAS<int,double> blas_;
    Teuchos::LAPACK<int,double> lapack_;

    std::vector<double> lhsBB_;
    std::vector<double> lhsIB_;
    std::vector<double> lhsBI_;
    std::vector<double> lhsII_;
    std::vector<double> rhsI_;
    std::vector<int> ipiv_;
    int nb_;
    int ni_;
    int ne_;

  };

} // namespace nalu
} // namespace Sierra

#endif
