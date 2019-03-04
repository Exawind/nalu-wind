/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporatlion.                                   */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#include <element_promotion/ElementCondenser.h>

#include <element_promotion/QuadratureRule.h>
#include <NaluEnv.h>

#include <stk_util/util/ReportHandler.hpp>
#include <Teuchos_RCP.hpp>

#include <tuple>
#include <element_promotion/ElementDescription.h>

namespace sierra {
namespace nalu {

//==========================================================================
// Class Definition
//==========================================================================
// ElementReducer - Condenses out interior degrees of freedom for
// linear elliptic problems (i.e. the ppe)
//==========================================================================
ElementCondenser::ElementCondenser(const ElementDescription& elem)
: blas_(Teuchos::BLAS<int,double>()),
  lapack_(Teuchos::LAPACK<int,double>())
{
  ne_ = elem.nodesPerElement;
  ni_ = std::pow(elem.polyOrder-1, elem.dimension);
  nb_ = ne_ - ni_;

  lhsBB_.resize(nb_ * nb_);
  lhsBI_.resize(nb_ * ni_);
  lhsIB_.resize(ni_ * nb_);
  lhsII_.resize(ni_ * ni_);
  ipiv_.resize(ni_ * ni_);
  rhsI_.resize(ni_);
}
//--------------------------------------------------------------------------
template <typename Scalar>
void mat_chunk(
  const Scalar* mat,
  int nrows,
  Scalar* chunk,
  int istart,
  int jstart,
  int iend,
  int jend)
{
  // copies matrix section of column major square matrix into contiguous, row-major submatrix
  ThrowAssert(istart >= 0 && istart < iend);
  ThrowAssert(jstart >= 0 && jstart < jend);
  ThrowAssert(iend <= nrows);
  ThrowAssert(jend <= nrows);

  int jj = 0;
  for (int j = jstart; j < jend; ++j) {
    for (int i = istart; i < iend; ++i, ++jj) {
      chunk[jj] = mat[i*nrows + j];
    }
  }
}
//--------------------------------------------------------------------------
void
ElementCondenser::condense(
  double* lhs, const double* rhs,
  double* b_lhs, double* b_rhs)
{
  // Computes the condensed left/right-hand sides for the high-order matrices
  // Split matrix into four contiguous chunks assuming a column-major "lhs"

  // NOTE: The nodes are numbered such that the nodes to be condensed out are
  // the last (p-1)^dim nodes in the matrix

  // boundary-boundary interaction
  mat_chunk(lhs, ne_,
    b_lhs,
    0, 0,
    nb_, nb_
  );

  // boundary-interior interaction
  mat_chunk(lhs, ne_,
    lhsBI_.data(),
    0, nb_,
    nb_, ne_
  );

  // interior-boundary interaction
  mat_chunk(lhs, ne_,
    lhsIB_.data(),
    nb_, 0,
    ne_, nb_
  );

  // interior-interior interaction
  mat_chunk(lhs, ne_,
    lhsII_.data(),
    nb_, nb_,
    ne_, ne_
  );

  // boundary terms for rhs vector
  for (int j = 0; j < nb_; ++j) {
    b_rhs[j] = rhs[j];
  }

  // interior terms for rhs vector
  int jj = 0;
  for (int j = nb_; j < ne_; ++j, ++jj) {
    rhsI_[jj] = rhs[j];
  }

  // compute LU decomposition of lhsII
  int info = 0;
  lapack_.GETRF(ni_,ni_,lhsII_.data(), ni_, ipiv_.data(), &info);
  ThrowAssert(info == 0);

  // solve for the effect of the interior on the boundary matrix L_II^-1 L_IB
  lapack_.GETRS('N',ni_,nb_,lhsII_.data(), ni_, ipiv_.data(), lhsIB_.data(), ni_, &info);
  ThrowAssert(info == 0);

  // apply modification to boundary-boundary matrix (L_BB - L_BI L_II^-1 L_IB)
  blas_.GEMM(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
    nb_, nb_, ni_,
    -1.0,
    lhsBI_.data(), nb_,
    lhsIB_.data(), ni_,
    +1.0,
    b_lhs, nb_
  );

  // compute interior effect on boundary rhs vector  L_II^-1 f_I
  lapack_.GETRS('N', ni_, 1, lhsII_.data(), ni_, ipiv_.data(), rhsI_.data(), ni_, &info);
  ThrowAssert(info == 0);

  // apply f_b - L_BI L_II^-1 f_I
  blas_.GEMV(Teuchos::NO_TRANS,
    nb_, ni_,
    -1.0,
    lhsBI_.data(), nb_,
    rhsI_.data(), 1,
    +1.0,
    b_rhs, 1
  );
}
//--------------------------------------------------------------------------
void ElementCondenser::compute_interior_update(
  double* lhs, const double* rhs,
  const double* /* boundary_values */, double* delta_interior_values)
{
  // Computes an update to the interior values given the updated rhs vector
  // and the elemental linear system
  mat_chunk(lhs, ne_,
    lhsII_.data(),
    nb_, nb_,
    ne_, ne_
  );

  int jj = 0;
  for (int j = nb_; j < ne_; ++j, ++jj) {
    delta_interior_values[jj] = rhs[j];
  }

  // solve for update L_II^-1 rhs_I
  int info = 0;
  lapack_.GESV(ni_, 1, lhsII_.data(), ni_, ipiv_.data(), delta_interior_values, ni_, &info);
  ThrowAssert(info == 0);
}

} // namespace nalu
} // namespace Sierra
