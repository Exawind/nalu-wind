/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef EIGENDECOMPOSITION_H
#define EIGENDECOMPOSITION_H

#include <SimdInterface.h>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

class Realm;
namespace EigenDecomposition
{
  void sym_diagonalize(const double (&A)[2][2], double (&Q)[2][2], double (&D)[2][2]);
  void sym_diagonalize(const double (&A)[3][3], double (&Q)[3][3], double (&D)[3][3]);
  void reconstruct_matrix_from_decomposition(const double (&D)[2][2], const double (&Q)[2][2], double (&A)[2][2]);
  void reconstruct_matrix_from_decomposition(const double (&D)[3][3], const double (&Q)[3][3], double (&A)[3][3]);
  void matrix_matrix_multiply(const double (&D)[2][2], const double (&Q)[2][2], double (&A)[2][2]);
  void matrix_matrix_multiply(const double (&D)[3][3], const double (&Q)[3][3], double (&A)[3][3]);

#ifndef STK_HAVE_NO_SIMD
  // Simd solvers
  void sym_diagonalize(const DoubleType (&A)[2][2], DoubleType (&Q)[2][2], DoubleType (&D)[2][2]);
  void sym_diagonalize(const DoubleType (&A)[3][3], DoubleType (&Q)[3][3], DoubleType (&D)[3][3]);
  void reconstruct_matrix_from_decomposition(const DoubleType (&D)[2][2], const DoubleType (&Q)[2][2], DoubleType (&A)[2][2]);
  void reconstruct_matrix_from_decomposition(const DoubleType (&D)[3][3], const DoubleType (&Q)[3][3], DoubleType (&A)[3][3]);
  void matrix_matrix_multiply(const DoubleType (&D)[2][2], const DoubleType (&Q)[2][2], DoubleType (&A)[2][2]);
  void matrix_matrix_multiply(const DoubleType (&D)[3][3], const DoubleType (&Q)[3][3], DoubleType (&A)[3][3]);
#endif

}

} // namespace nalu
} // namespace sierra

#endif
