// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EIGENDECOMPOSITION_H
#define EIGENDECOMPOSITION_H

#include <FieldTypeDef.h>
#include <SimdInterface.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

class Realm;
namespace EigenDecomposition {

//--------------------------------------------------------------------------
//-------- symmetric diagonalize (2D) --------------------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
sym_diagonalize(const T (&A)[2][2], T (&Q)[2][2], T (&D)[2][2])
{
  // Note that A must be symmetric here
  const T trace = A[0][0] + A[1][1];
  const T det = A[0][0] * A[1][1] - A[0][1] * A[1][0];

  // calculate eigenvalues
  D[0][0] = stk::math::if_then_else(
    A[1][0] == 0.0, A[0][0],
    trace / 2.0 + stk::math::sqrt(trace * trace / 4.0 - det));
  D[1][1] = stk::math::if_then_else(
    A[1][0] == 0.0, A[1][1],
    trace / 2.0 - stk::math::sqrt(trace * trace / 4.0 - det));
  D[0][1] = 0.0;
  D[1][0] = 0.0;

  // calculate first eigenvector
  Q[0][0] = -A[0][1];
  Q[1][0] = A[0][0] - D[0][0];

  T norm = stk::math::sqrt(Q[0][0] * Q[0][0] + Q[1][0] * Q[1][0]);
  Q[0][0] = Q[0][0] / norm;
  Q[1][0] = Q[1][0] / norm;

  // calculate second eigenvector
  Q[0][1] = -A[1][1] + D[1][1];
  Q[1][1] = A[1][0];

  norm = stk::math::sqrt(Q[0][1] * Q[0][1] + Q[1][1] * Q[1][1]);
  Q[0][1] = Q[0][1] / norm;
  Q[1][1] = Q[1][1] / norm;

  // special case when off diagonal entries were 0, we already had a diagonal
  // matrix
  Q[0][0] = stk::math::if_then_else(A[1][0] == 0.0, 1.0, Q[0][0]);
  Q[0][1] = stk::math::if_then_else(A[1][0] == 0.0, 0.0, Q[0][1]);
  Q[1][0] = stk::math::if_then_else(A[1][0] == 0.0, 0.0, Q[1][0]);
  Q[1][1] = stk::math::if_then_else(A[1][0] == 0.0, 1.0, Q[1][1]);
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 2D ---------------------------------------
//--------------------------------------------------------------------------
template <class T>
void
matrix_matrix_multiply(const T (&A)[2][2], const T (&B)[2][2], T (&C)[2][2])
{
  // C = A*B
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      T sum = 0;
      for (int k = 0; k < 2; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 2D ------------------------
//--------------------------------------------------------------------------
template <class T>
void
reconstruct_matrix_from_decomposition(
  const T (&D)[2][2], const T (&Q)[2][2], T (&A)[2][2])
{
  // A = Q*D*QT
  T QT[2][2];
  T B[2][2];

  // compute QT
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  // mat-vec, B = Q*D
  matrix_matrix_multiply(Q, D, B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B, QT, A);
}
//--------------------------------------------------------------------------
//-------- symmetric diagonalize (3D) --------------------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
sym_diagonalize(const T (&A)[3][3], T (&Q)[3][3], T (&D)[3][3])
{
  /*
    obtained from:
    http://stackoverflow.com/questions/4372224/
    fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition

    A must be a symmetric matrix.
    returns Q and D such that
    Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
  */

  const int maxsteps = 24;
  T o[3], m[3];
  T q[4] = {0.0, 0.0, 0.0, 1.0};
  T jr[4];
  T sqw, sqx, sqy, sqz;
  T tmp1, tmp2, mq;
  T AQ[3][3];
  T thet, sgn, t, c;

  T jrL, oLarge, dDiff;

  for (int i = 0; i < maxsteps; ++i) {
    // quat to matrix
    sqx = q[0] * q[0];
    sqy = q[1] * q[1];
    sqz = q[2] * q[2];
    sqw = q[3] * q[3];
    Q[0][0] = (sqx - sqy - sqz + sqw);
    Q[1][1] = (-sqx + sqy - sqz + sqw);
    Q[2][2] = (-sqx - sqy + sqz + sqw);
    tmp1 = q[0] * q[1];
    tmp2 = q[2] * q[3];
    Q[1][0] = 2.0 * (tmp1 + tmp2);
    Q[0][1] = 2.0 * (tmp1 - tmp2);
    tmp1 = q[0] * q[2];
    tmp2 = q[1] * q[3];
    Q[2][0] = 2.0 * (tmp1 - tmp2);
    Q[0][2] = 2.0 * (tmp1 + tmp2);
    tmp1 = q[1] * q[2];
    tmp2 = q[0] * q[3];
    Q[2][1] = 2.0 * (tmp1 + tmp2);
    Q[1][2] = 2.0 * (tmp1 - tmp2);

    // AQ = A * Q
    AQ[0][0] = Q[0][0] * A[0][0] + Q[1][0] * A[0][1] + Q[2][0] * A[0][2];
    AQ[0][1] = Q[0][1] * A[0][0] + Q[1][1] * A[0][1] + Q[2][1] * A[0][2];
    AQ[0][2] = Q[0][2] * A[0][0] + Q[1][2] * A[0][1] + Q[2][2] * A[0][2];
    AQ[1][0] = Q[0][0] * A[0][1] + Q[1][0] * A[1][1] + Q[2][0] * A[1][2];
    AQ[1][1] = Q[0][1] * A[0][1] + Q[1][1] * A[1][1] + Q[2][1] * A[1][2];
    AQ[1][2] = Q[0][2] * A[0][1] + Q[1][2] * A[1][1] + Q[2][2] * A[1][2];
    AQ[2][0] = Q[0][0] * A[0][2] + Q[1][0] * A[1][2] + Q[2][0] * A[2][2];
    AQ[2][1] = Q[0][1] * A[0][2] + Q[1][1] * A[1][2] + Q[2][1] * A[2][2];
    AQ[2][2] = Q[0][2] * A[0][2] + Q[1][2] * A[1][2] + Q[2][2] * A[2][2];
    // D = Qt * AQ
    D[0][0] = AQ[0][0] * Q[0][0] + AQ[1][0] * Q[1][0] + AQ[2][0] * Q[2][0];
    D[0][1] = AQ[0][0] * Q[0][1] + AQ[1][0] * Q[1][1] + AQ[2][0] * Q[2][1];
    D[0][2] = AQ[0][0] * Q[0][2] + AQ[1][0] * Q[1][2] + AQ[2][0] * Q[2][2];
    D[1][0] = AQ[0][1] * Q[0][0] + AQ[1][1] * Q[1][0] + AQ[2][1] * Q[2][0];
    D[1][1] = AQ[0][1] * Q[0][1] + AQ[1][1] * Q[1][1] + AQ[2][1] * Q[2][1];
    D[1][2] = AQ[0][1] * Q[0][2] + AQ[1][1] * Q[1][2] + AQ[2][1] * Q[2][2];
    D[2][0] = AQ[0][2] * Q[0][0] + AQ[1][2] * Q[1][0] + AQ[2][2] * Q[2][0];
    D[2][1] = AQ[0][2] * Q[0][1] + AQ[1][2] * Q[1][1] + AQ[2][2] * Q[2][1];
    D[2][2] = AQ[0][2] * Q[0][2] + AQ[1][2] * Q[1][2] + AQ[2][2] * Q[2][2];
    o[0] = D[1][2];
    o[1] = D[0][2];
    o[2] = D[0][1];
    m[0] = stk::math::abs(o[0]);
    m[1] = stk::math::abs(o[1]);
    m[2] = stk::math::abs(o[2]);

    // index of largest element of offdiag
    oLarge =
      stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), D[1][2], 0.0);
    oLarge =
      stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), D[0][2], oLarge);
    oLarge =
      stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), D[0][1], oLarge);

    dDiff = stk::math::if_then_else(
      (m[0] > m[1]) && (m[0] > m[2]), D[2][2] - D[1][1], 0.0);
    dDiff = stk::math::if_then_else(
      (m[1] > m[2]) && (m[1] > m[0]), D[0][0] - D[2][2], dDiff);
    dDiff = stk::math::if_then_else(
      (m[2] > m[1]) && (m[2] > m[0]), D[1][1] - D[0][0], dDiff);

    // if oLarge == 0.0, then we are already diagonal
    // we need to be able to divide by thet, so set to 1.0 temporarily and
    // catch c at the end and correct it to 1 to handle the diagonal case
    thet =
      stk::math::if_then_else(oLarge == 0.0, 1.0, (dDiff) / (2.0 * oLarge));
    sgn = stk::math::if_then_else(thet > 0.0, 1.0, -1.0);
    thet = thet * sgn;
    // sign(T)/(|T|+sqrt(T^2+1))
    t = stk::math::if_then_else(
      thet < 1.E6, sgn / (thet + stk::math::sqrt(thet * thet + 1.0)),
      0.5 * sgn / thet);
    c = stk::math::if_then_else(
      oLarge == 0.0, 1.0, 1.0 / stk::math::sqrt(t * t + 1.0));

    // using 1/2 angle identity sin(a/2) = std::sqrt((1-cos(a))/2)
    // -1.0 since our quat-to-matrix convention was for v*M instead of M*v
    jr[0] = stk::math::if_then_else(
      (m[0] > m[1]) && (m[0] > m[2]),
      -1.0 * (sgn * stk::math::sqrt((1.0 - c) / 2.0)), 0.0);
    jr[1] = stk::math::if_then_else(
      (m[1] > m[2]) && (m[1] > m[0]),
      -1.0 * (sgn * stk::math::sqrt((1.0 - c) / 2.0)), 0.0);
    jr[2] = stk::math::if_then_else(
      (m[2] > m[1]) && (m[2] > m[0]),
      -1.0 * (sgn * stk::math::sqrt((1.0 - c) / 2.0)), 0.0);

    jrL = stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), jr[0], 0.0);
    jrL = stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), jr[1], jrL);
    jrL = stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), jr[2], jrL);

    jr[3] = stk::math::sqrt(1.0f - jrL * jrL);

    const auto check_one = jr[3] == 1.0;
    const bool exit_now = stk::simd::are_all(check_one);
    if (exit_now) {
      break; // reached limits of floating point precision
    }

    q[0] = (q[3] * jr[0] + q[0] * jr[3] + q[1] * jr[2] - q[2] * jr[1]);
    q[1] = (q[3] * jr[1] - q[0] * jr[2] + q[1] * jr[3] + q[2] * jr[0]);
    q[2] = (q[3] * jr[2] + q[0] * jr[1] - q[1] * jr[0] + q[2] * jr[3]);
    q[3] = (q[3] * jr[3] - q[0] * jr[0] - q[1] * jr[1] - q[2] * jr[2]);
    mq = stk::math::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0] /= mq;
    q[1] /= mq;
    q[2] /= mq;
    q[3] /= mq;
  }
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 3D ---------------------------------------
//--------------------------------------------------------------------------
template <class T>
void
matrix_matrix_multiply(const T (&A)[3][3], const T (&B)[3][3], T (&C)[3][3])
{
  // C = A*B
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      T sum = 0;
      for (int k = 0; k < 3; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 3D ------------------------
//--------------------------------------------------------------------------
template <class T>
void
reconstruct_matrix_from_decomposition(
  const T (&D)[3][3], const T (&Q)[3][3], T (&A)[3][3])
{
  // A = Q*D*QT
  T QT[3][3];
  T B[3][3];

  // compute QT
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  // mat-vec, B = Q*D
  matrix_matrix_multiply(Q, D, B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B, QT, A);
}

//--------------------------------------------------------------------------
//------------------ unsym_matrix_force_sym_3D -----------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
unsym_matrix_force_sym(T (&A)[3][3], T (&Q)[3][3], T (&D)[3][3])
{

  // force symmetry force
  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      A[i][j] = (A[i][j] + A[j][i]) / 2.0;
      A[j][i] = A[i][j];
    }
  }

  // then call symmetric diagonalize
  sym_diagonalize(A, Q, D);
}

//--------------------------------------------------------------------------
//------------------ general_eigenvalues_3D -----------------------------
//--------------------------------------------------------------------------
template <class T>
KOKKOS_FUNCTION void
general_eigenvalues(T (&A)[3][3], T (&Q)[3][3], T (&D)[3][3])
{

  const T pi = stk::math::acos(-1.0);
  const T machEps = std::numeric_limits<T>::min();

  // Characteristic equation for A is ax^3 + bx^2 + cx + d = 0 where x are the
  // eigenvalues and a = 1, b = -trA, c = coFacA, d = -detA
  const T trA = A[0][0] + A[1][1] + A[2][2];
  const T detA = A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] - A[0][2] * A[1][1] * A[2][0];
  const T coFacA = A[0][0] * A[1][1] - A[0][1] * A[1][0] + A[1][1] * A[2][2] -
                   A[1][2] * A[2][1] + A[0][0] * A[2][2] - A[0][2] * A[2][0];

  // Check to make sure all eigenvalues are real
  // discriminant = (bc)^2 - 4ac^3 -4b^3d -27a^2d^2 + 18abcd where
  // a = 1, b = -trA, c = coFacA, d = -detA
  const T disc = trA * trA * coFacA * coFacA - 4.0 * coFacA * coFacA * coFacA -
                 4.0 * trA * trA * trA * detA - 27.0 * detA * detA +
                 18.0 * trA * coFacA * detA;

  const auto check_one = disc < -machEps;
  const bool exit_now = stk::simd::are_all(check_one);
  if (exit_now) {
#if !defined(KOKKOS_ENABLE_GPU)
    NaluEnv::self().naluOutput()
      << "Error, complex eigenvalues in EigenDecomposition::general_eigenvalues"
      << disc << "([[" << A[0][0] << "," << A[0][1] << "," << A[0][2] << "],["
      << A[1][0] << "," << A[1][1] << "," << A[1][2] << "],[" << A[2][0] << ","
      << A[2][1] << "," << A[2][2] << "]])" << std::endl;
    throw std::runtime_error(
      "ERROR, complex eigenvalues in EigenDecomposition::general_eigenvalues");

#else
    ThrowErrorMsgDevice(
      "ERROR, complex eigenvalues in EigenDecomposition::general_eigenvalues");
#endif
  }

  // Convert to depressed cubic (substitute x = t - b/3a = t + trA/3)
  // This leads to cubic: t^3 + pt + q  where the linear and constant
  // coefficient are defined as below
  const T linCoef = coFacA - trA * trA / 3.0;
  const T constCoef = coFacA * trA / 3.0 - 2.0 * trA * trA * trA / 27.0 - detA;

  // Solve roots of depressed cubic polynomial analytically (Francois Viete
  // formula)
  const T t1 =
    2.0 * stk::math::sqrt(-linCoef / 3.0) *
    stk::math::cos(
      stk::math::acos(
        3.0 * constCoef * stk::math::sqrt(-3.0 / linCoef) / (2.0 * linCoef)) /
      3.0);
  const T t2 =
    2.0 * stk::math::sqrt(-linCoef / 3.0) *
    stk::math::cos(
      stk::math::acos(
        3.0 * constCoef * stk::math::sqrt(-3.0 / linCoef) / (2.0 * linCoef)) /
        3.0 -
      2.0 * pi / 3.0);
  const T t3 =
    2.0 * stk::math::sqrt(-linCoef / 3.0) *
    stk::math::cos(
      stk::math::acos(
        3.0 * constCoef * stk::math::sqrt(-3.0 / linCoef) / (2.0 * linCoef)) /
        3.0 -
      4.0 * pi / 3.0);

  // Convert roots of depressed polynomial back to the eigenvalues
  D[0][0] = t1 + trA / 3.0;
  D[1][1] = t2 + trA / 3.0;
  D[2][2] = t3 + trA / 3.0;

  // Zero out Q since this only returns eigenvalues
  Q[0][0] = Q[0][1] = Q[0][2] = Q[1][0] = Q[1][1] = Q[1][2] = Q[2][0] =
    Q[2][1] = Q[2][2] = 0.0;
}

} // namespace EigenDecomposition

} // namespace nalu
} // namespace sierra

#endif
