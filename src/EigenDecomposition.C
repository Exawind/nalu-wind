/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <EigenDecomposition.h>
#include <NaluParsing.h>

#include <SimdInterface.h>

// basic c++
#include <stdexcept>

namespace sierra{
namespace nalu{

//--------------------------------------------------------------------------
//-------- symmetric diagonalize (2D) --------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::sym_diagonalize(
  const double (&A)[2][2], double (&Q)[2][2], double (&D)[2][2])
{
  // Note that A must be symmetric here
  const double trace = A[0][0] + A[1][1];
  const double det = A[0][0]*A[1][1] - A[0][1]*A[1][0];

  // calculate eigenvalues
  D[0][0] = (A[1][0] == 0.0) ? A[0][0] : trace/2.0 + std::sqrt(trace*trace/4.0 - det);
  D[1][1] = (A[1][0] == 0.0) ? A[1][1] : trace/2.0 - std::sqrt(trace*trace/4.0 - det);
  D[0][1] = 0.0;
  D[1][0] = 0.0;

  //calculate first eigenvector
  Q[0][0] = -A[0][1];
  Q[1][0] =  A[0][0] - D[0][0];

  double norm = std::sqrt(Q[0][0]*Q[0][0] + Q[1][0]*Q[1][0]);
  Q[0][0] = Q[0][0]/norm;
  Q[1][0] = Q[1][0]/norm;

  // calculate second eigenvector
  Q[0][1] = -A[1][1] + D[1][1];
  Q[1][1] =  A[1][0];

  norm = std::sqrt(Q[0][1]*Q[0][1] + Q[1][1]*Q[1][1]);
  Q[0][1] = Q[0][1]/norm;
  Q[1][1] = Q[1][1]/norm;

  // special case when off diagonal entries were 0, we already had a diagonal matrix
  Q[0][0] = (A[1][0] == 0.0) ? 1.0 : Q[0][0];
  Q[0][1] = (A[1][0] == 0.0) ? 0.0 : Q[0][1];
  Q[1][0] = (A[1][0] == 0.0) ? 0.0 : Q[1][0];
  Q[1][1] = (A[1][0] == 0.0) ? 1.0 : Q[1][1];
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 2D ------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::reconstruct_matrix_from_decomposition(
  const double (&D)[2][2], const double (&Q)[2][2], double (&A)[2][2])
{
  // A = Q*D*QT
  double QT[2][2];
  double B[2][2];

  // compute QT
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  //mat-vec, B = Q*D
  matrix_matrix_multiply(Q,D,B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B,QT,A);
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 2D ---------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::matrix_matrix_multiply(
  const double (&A)[2][2], const double (&B)[2][2], double (&C)[2][2])
{
  //C = A*B
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double sum = 0;
      for (int k = 0; k < 2; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- symmetric diagonalize (3D) --------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::sym_diagonalize(
  const double (&A)[3][3], double (&Q)[3][3], double (&D)[3][3])
{
  /*
    obtained from: 
    http://stackoverflow.com/questions/4372224/
    fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition

    A must be a symmetric matrix.
    returns Q and D such that 
    Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
  */

  const int maxsteps=24;
  int k0, k1, k2;
  double o[3], m[3];
  double q [4] = {0.0,0.0,0.0,1.0};
  double jr[4];
  double sqw, sqx, sqy, sqz;
  double tmp1, tmp2, mq;
  double AQ[3][3];
  double thet, sgn, t, c;
  for(int i=0;i < maxsteps;++i) {
    // quat to matrix
    sqx      = q[0]*q[0];
    sqy      = q[1]*q[1];
    sqz      = q[2]*q[2];
    sqw      = q[3]*q[3];
    Q[0][0]  = ( sqx - sqy - sqz + sqw);
    Q[1][1]  = (-sqx + sqy - sqz + sqw);
    Q[2][2]  = (-sqx - sqy + sqz + sqw);
    tmp1     = q[0]*q[1];
    tmp2     = q[2]*q[3];
    Q[1][0]  = 2.0 * (tmp1 + tmp2);
    Q[0][1]  = 2.0 * (tmp1 - tmp2);
    tmp1     = q[0]*q[2];
    tmp2     = q[1]*q[3];
    Q[2][0]  = 2.0 * (tmp1 - tmp2);
    Q[0][2]  = 2.0 * (tmp1 + tmp2);
    tmp1     = q[1]*q[2];
    tmp2     = q[0]*q[3];
    Q[2][1]  = 2.0 * (tmp1 + tmp2);
    Q[1][2]  = 2.0 * (tmp1 - tmp2);

    // AQ = A * Q
    AQ[0][0] = Q[0][0]*A[0][0]+Q[1][0]*A[0][1]+Q[2][0]*A[0][2];
    AQ[0][1] = Q[0][1]*A[0][0]+Q[1][1]*A[0][1]+Q[2][1]*A[0][2];
    AQ[0][2] = Q[0][2]*A[0][0]+Q[1][2]*A[0][1]+Q[2][2]*A[0][2];
    AQ[1][0] = Q[0][0]*A[0][1]+Q[1][0]*A[1][1]+Q[2][0]*A[1][2];
    AQ[1][1] = Q[0][1]*A[0][1]+Q[1][1]*A[1][1]+Q[2][1]*A[1][2];
    AQ[1][2] = Q[0][2]*A[0][1]+Q[1][2]*A[1][1]+Q[2][2]*A[1][2];
    AQ[2][0] = Q[0][0]*A[0][2]+Q[1][0]*A[1][2]+Q[2][0]*A[2][2];
    AQ[2][1] = Q[0][1]*A[0][2]+Q[1][1]*A[1][2]+Q[2][1]*A[2][2];
    AQ[2][2] = Q[0][2]*A[0][2]+Q[1][2]*A[1][2]+Q[2][2]*A[2][2];
    // D = Qt * AQ
    D[0][0] = AQ[0][0]*Q[0][0]+AQ[1][0]*Q[1][0]+AQ[2][0]*Q[2][0];
    D[0][1] = AQ[0][0]*Q[0][1]+AQ[1][0]*Q[1][1]+AQ[2][0]*Q[2][1];
    D[0][2] = AQ[0][0]*Q[0][2]+AQ[1][0]*Q[1][2]+AQ[2][0]*Q[2][2];
    D[1][0] = AQ[0][1]*Q[0][0]+AQ[1][1]*Q[1][0]+AQ[2][1]*Q[2][0];
    D[1][1] = AQ[0][1]*Q[0][1]+AQ[1][1]*Q[1][1]+AQ[2][1]*Q[2][1];
    D[1][2] = AQ[0][1]*Q[0][2]+AQ[1][1]*Q[1][2]+AQ[2][1]*Q[2][2];
    D[2][0] = AQ[0][2]*Q[0][0]+AQ[1][2]*Q[1][0]+AQ[2][2]*Q[2][0];
    D[2][1] = AQ[0][2]*Q[0][1]+AQ[1][2]*Q[1][1]+AQ[2][2]*Q[2][1];
    D[2][2] = AQ[0][2]*Q[0][2]+AQ[1][2]*Q[1][2]+AQ[2][2]*Q[2][2];
    o[0]    = D[1][2];
    o[1]    = D[0][2];
    o[2]    = D[0][1];
    m[0]    = std::abs(o[0]);
    m[1]    = std::abs(o[1]);
    m[2]    = std::abs(o[2]);

    k0      = (m[0] > m[1] && m[0] > m[2])?0: (m[1] > m[2])? 1 : 2; // index of largest element of offdiag
    k1      = (k0+1)%3;
    k2      = (k0+2)%3;
    if (o[k0]==0.0) {
      break;  // diagonal already
    }
    thet    = (D[k2][k2]-D[k1][k1])/(2.0*o[k0]);
    sgn     = (thet > 0.0)?1.0:-1.0;
    thet   *= sgn; // make it positive
    t       = sgn /(thet +((thet < 1.E6)? std::sqrt(thet*thet+1.0):thet)) ; // sign(T)/(|T|+sqrt(T^2+1))
    c       = 1.0/std::sqrt(t*t+1.0); //  c= 1/(t^2+1) , t=s/c 
    if(c==1.0) {
      break;  // no room for improvement - reached machine precision.
    }
    jr[0 ]  = jr[1] = jr[2] = jr[3] = 0.0;
    jr[k0]  = sgn*std::sqrt((1.0-c)/2.0);  // using 1/2 angle identity sin(a/2) = std::sqrt((1-cos(a))/2)  
    jr[k0] *= -1.0; // since our quat-to-matrix convention was for v*M instead of M*v
    jr[3 ]  = std::sqrt(1.0f - jr[k0] * jr[k0]);
    if(jr[3]==1.0) {
      break; // reached limits of floating point precision
    }
    q[0]    = (q[3]*jr[0] + q[0]*jr[3] + q[1]*jr[2] - q[2]*jr[1]);
    q[1]    = (q[3]*jr[1] - q[0]*jr[2] + q[1]*jr[3] + q[2]*jr[0]);
    q[2]    = (q[3]*jr[2] + q[0]*jr[1] - q[1]*jr[0] + q[2]*jr[3]);
    q[3]    = (q[3]*jr[3] - q[0]*jr[0] - q[1]*jr[1] - q[2]*jr[2]);
    mq      = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0]   /= mq;
    q[1]   /= mq;
    q[2]   /= mq;
    q[3]   /= mq;
  }
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 3D ------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::reconstruct_matrix_from_decomposition(
  const double (&D)[3][3], const double (&Q)[3][3], double (&A)[3][3])
{
  // A = Q*D*QT
  double QT[3][3];
  double B[3][3];

  // compute QT
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  //mat-vec, B = Q*D
  matrix_matrix_multiply(Q,D,B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B,QT,A);
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 3D ---------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::matrix_matrix_multiply(
  const double (&A)[3][3], const double (&B)[3][3], double (&C)[3][3])
{
  //C = A*B
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double sum = 0;
      for (int k = 0; k < 3; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

#ifndef STK_HAVE_NO_SIMD
//--------------------------------------------------------------------------
//------------- SIMD solvers -----------------------------------------------
//--------------------------------------------------------------------------

//--------------------------------------------------------------------------
//-------- symmetric diagonalize (2D) --------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::sym_diagonalize(
  const DoubleType (&A)[2][2], DoubleType (&Q)[2][2], DoubleType (&D)[2][2])
{
  // Note that A must be symmetric here
  const DoubleType trace = A[0][0] + A[1][1];
  const DoubleType det = A[0][0]*A[1][1] - A[0][1]*A[1][0];

  // calculate eigenvalues
  D[0][0] = stk::math::if_then_else(A[1][0] == 0.0, A[0][0], trace/2.0 + stk::math::sqrt(trace*trace/4.0 - det));
  D[1][1] = stk::math::if_then_else(A[1][0] == 0.0, A[1][1], trace/2.0 - stk::math::sqrt(trace*trace/4.0 - det));
  D[0][1] = 0.0;
  D[1][0] = 0.0;

  //calculate first eigenvector
  Q[0][0] = -A[0][1];
  Q[1][0] =  A[0][0] - D[0][0];

  DoubleType norm = stk::math::sqrt(Q[0][0]*Q[0][0] + Q[1][0]*Q[1][0]);
  Q[0][0] = Q[0][0]/norm;
  Q[1][0] = Q[1][0]/norm;

  // calculate second eigenvector
  Q[0][1] = -A[1][1] + D[1][1];
  Q[1][1] =  A[1][0];

  norm = stk::math::sqrt(Q[0][1]*Q[0][1] + Q[1][1]*Q[1][1]);
  Q[0][1] = Q[0][1]/norm;
  Q[1][1] = Q[1][1]/norm;

  // special case when off diagonal entries were 0, we already had a diagonal matrix
  Q[0][0] = stk::math::if_then_else(A[1][0] == 0.0, 1.0, Q[0][0]);
  Q[0][1] = stk::math::if_then_else(A[1][0] == 0.0, 0.0, Q[0][1]);
  Q[1][0] = stk::math::if_then_else(A[1][0] == 0.0, 0.0, Q[1][0]);
  Q[1][1] = stk::math::if_then_else(A[1][0] == 0.0, 1.0, Q[1][1]);
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 2D ------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::reconstruct_matrix_from_decomposition(
  const DoubleType (&D)[2][2], const DoubleType (&Q)[2][2], DoubleType (&A)[2][2])
{
  // A = Q*D*QT
  DoubleType QT[2][2];
  DoubleType B[2][2];

  // compute QT
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  //mat-vec, B = Q*D
  matrix_matrix_multiply(Q,D,B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B,QT,A);
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 2D ---------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::matrix_matrix_multiply(
  const DoubleType (&A)[2][2], const DoubleType (&B)[2][2], DoubleType (&C)[2][2])
{
  //C = A*B
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      DoubleType sum = 0;
      for (int k = 0; k < 2; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//--------------------------------------------------------------------------
//-------- symmetric diagonalize (3D) --------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::sym_diagonalize(
  const DoubleType (&A)[3][3], DoubleType (&Q)[3][3], DoubleType (&D)[3][3])
{
  /*
    obtained from: 
    http://stackoverflow.com/questions/4372224/
    fast-method-for-computing-3x3-symmetric-matrix-spectral-decomposition

    A must be a symmetric matrix.
    returns Q and D such that 
    Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
  */

  const int maxsteps=24;
  DoubleType o[3], m[3];
  DoubleType q [4] = {0.0,0.0,0.0,1.0};
  DoubleType jr[4];
  DoubleType sqw, sqx, sqy, sqz;
  DoubleType tmp1, tmp2, mq;
  DoubleType AQ[3][3];
  DoubleType thet, sgn, t, c;

  DoubleType jrL, oLarge, lIndex, dDiff;

  for(int i = 0; i < maxsteps; ++i) {
    // quat to matrix
    sqx      = q[0]*q[0];
    sqy      = q[1]*q[1];
    sqz      = q[2]*q[2];
    sqw      = q[3]*q[3];
    Q[0][0]  = ( sqx - sqy - sqz + sqw);
    Q[1][1]  = (-sqx + sqy - sqz + sqw);
    Q[2][2]  = (-sqx - sqy + sqz + sqw);
    tmp1     = q[0]*q[1];
    tmp2     = q[2]*q[3];
    Q[1][0]  = 2.0 * (tmp1 + tmp2);
    Q[0][1]  = 2.0 * (tmp1 - tmp2);
    tmp1     = q[0]*q[2];
    tmp2     = q[1]*q[3];
    Q[2][0]  = 2.0 * (tmp1 - tmp2);
    Q[0][2]  = 2.0 * (tmp1 + tmp2);
    tmp1     = q[1]*q[2];
    tmp2     = q[0]*q[3];
    Q[2][1]  = 2.0 * (tmp1 + tmp2);
    Q[1][2]  = 2.0 * (tmp1 - tmp2);

    // AQ = A * Q
    AQ[0][0] = Q[0][0]*A[0][0]+Q[1][0]*A[0][1]+Q[2][0]*A[0][2];
    AQ[0][1] = Q[0][1]*A[0][0]+Q[1][1]*A[0][1]+Q[2][1]*A[0][2];
    AQ[0][2] = Q[0][2]*A[0][0]+Q[1][2]*A[0][1]+Q[2][2]*A[0][2];
    AQ[1][0] = Q[0][0]*A[0][1]+Q[1][0]*A[1][1]+Q[2][0]*A[1][2];
    AQ[1][1] = Q[0][1]*A[0][1]+Q[1][1]*A[1][1]+Q[2][1]*A[1][2];
    AQ[1][2] = Q[0][2]*A[0][1]+Q[1][2]*A[1][1]+Q[2][2]*A[1][2];
    AQ[2][0] = Q[0][0]*A[0][2]+Q[1][0]*A[1][2]+Q[2][0]*A[2][2];
    AQ[2][1] = Q[0][1]*A[0][2]+Q[1][1]*A[1][2]+Q[2][1]*A[2][2];
    AQ[2][2] = Q[0][2]*A[0][2]+Q[1][2]*A[1][2]+Q[2][2]*A[2][2];
    // D = Qt * AQ
    D[0][0] = AQ[0][0]*Q[0][0]+AQ[1][0]*Q[1][0]+AQ[2][0]*Q[2][0];
    D[0][1] = AQ[0][0]*Q[0][1]+AQ[1][0]*Q[1][1]+AQ[2][0]*Q[2][1];
    D[0][2] = AQ[0][0]*Q[0][2]+AQ[1][0]*Q[1][2]+AQ[2][0]*Q[2][2];
    D[1][0] = AQ[0][1]*Q[0][0]+AQ[1][1]*Q[1][0]+AQ[2][1]*Q[2][0];
    D[1][1] = AQ[0][1]*Q[0][1]+AQ[1][1]*Q[1][1]+AQ[2][1]*Q[2][1];
    D[1][2] = AQ[0][1]*Q[0][2]+AQ[1][1]*Q[1][2]+AQ[2][1]*Q[2][2];
    D[2][0] = AQ[0][2]*Q[0][0]+AQ[1][2]*Q[1][0]+AQ[2][2]*Q[2][0];
    D[2][1] = AQ[0][2]*Q[0][1]+AQ[1][2]*Q[1][1]+AQ[2][2]*Q[2][1];
    D[2][2] = AQ[0][2]*Q[0][2]+AQ[1][2]*Q[1][2]+AQ[2][2]*Q[2][2];
    o[0]    = D[1][2];
    o[1]    = D[0][2];
    o[2]    = D[0][1];
    m[0]    = stk::math::abs(o[0]);
    m[1]    = stk::math::abs(o[1]);
    m[2]    = stk::math::abs(o[2]);

    // index of largest element of offdiag
    oLarge  = stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), D[1][2], 0.0);
    oLarge  = stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), D[0][2], oLarge);
    oLarge  = stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), D[0][1], oLarge);

    dDiff   = stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), D[2][2] - D[1][1], 0.0);
    dDiff   = stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), D[0][0] - D[2][2], dDiff);
    dDiff   = stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), D[1][1] - D[0][0], dDiff);

    // if oLarge == 0.0, then we are already diagonal
    // we need to be able to divide by thet, so set to 1.0 temporarily and 
    // catch c at the end and correct it to 1 to handle the diagonal case
    thet = stk::math::if_then_else(oLarge == 0.0, 1.0, (dDiff)/(2.0*oLarge));
    sgn  = stk::math::if_then_else(thet > 0.0, 1.0, -1.0);
    thet = thet * sgn;
    // sign(T)/(|T|+sqrt(T^2+1))
    t    = stk::math::if_then_else(thet < 1.E6, sgn/(thet + stk::math::sqrt(thet*thet+1.0)), 0.5*sgn/thet);
    c    = stk::math::if_then_else(oLarge == 0.0, 1.0, 1.0/stk::math::sqrt(t*t+1.0));

    // using 1/2 angle identity sin(a/2) = std::sqrt((1-cos(a))/2)
    // -1.0 since our quat-to-matrix convention was for v*M instead of M*v
    jr[0] = stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), -1.0 * (sgn*stk::math::sqrt((1.0-c)/2.0)), 0.0); 
    jr[1] = stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), -1.0 * (sgn*stk::math::sqrt((1.0-c)/2.0)), 0.0);
    jr[2] = stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), -1.0 * (sgn*stk::math::sqrt((1.0-c)/2.0)), 0.0);

    jrL = stk::math::if_then_else((m[0] > m[1]) && (m[0] > m[2]), jr[0], 0.0);
    jrL = stk::math::if_then_else((m[1] > m[2]) && (m[1] > m[0]), jr[1], jrL);
    jrL = stk::math::if_then_else((m[2] > m[1]) && (m[2] > m[0]), jr[2], jrL);

    jr[3] = stk::math::sqrt(1.0f - jrL * jrL);

    // FIXME: Is there a better way to deal with the exit condition than having to hack to examine
    //        each object in the SIMD array.  Maybe some sort of max/min operation?
    int cnt = 0;
    for(int simdIndex=0; simdIndex<simdLen; ++simdIndex)
      if (stk::simd::get_data(jr[3], simdIndex) == 1.0) cnt++;

    if(cnt == simdLen) {
      break; // reached limits of floating point precision
    }

    q[0]    = (q[3]*jr[0] + q[0]*jr[3] + q[1]*jr[2] - q[2]*jr[1]);
    q[1]    = (q[3]*jr[1] - q[0]*jr[2] + q[1]*jr[3] + q[2]*jr[0]);
    q[2]    = (q[3]*jr[2] + q[0]*jr[1] - q[1]*jr[0] + q[2]*jr[3]);
    q[3]    = (q[3]*jr[3] - q[0]*jr[0] - q[1]*jr[1] - q[2]*jr[2]);
    mq      = stk::math::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    q[0]   /= mq;
    q[1]   /= mq;
    q[2]   /= mq;
    q[3]   /= mq;
  }
}

//--------------------------------------------------------------------------
//-------- reconstruct_matrix_from_decomposition 3D ------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::reconstruct_matrix_from_decomposition(
  const DoubleType (&D)[3][3], const DoubleType (&Q)[3][3], DoubleType (&A)[3][3])
{
  // A = Q*D*QT
  DoubleType QT[3][3];
  DoubleType B[3][3];

  // compute QT
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      QT[j][i] = Q[i][j];
    }
  }
  //mat-vec, B = Q*D
  matrix_matrix_multiply(Q,D,B);

  // mat-vec, A = (Q*D)*QT = B*QT
  matrix_matrix_multiply(B,QT,A);
}

//--------------------------------------------------------------------------
//-------- matrix_matrix_multiply 3D ---------------------------------------
//--------------------------------------------------------------------------
void
EigenDecomposition::matrix_matrix_multiply(
  const DoubleType (&A)[3][3], const DoubleType (&B)[3][3], DoubleType (&C)[3][3])
{
  //C = A*B
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      DoubleType sum = 0;
      for (int k = 0; k < 3; ++k) {
        sum = sum + A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}
#endif // STK_HAVE_NO_SIMD

} // namespace nalu
} // namespace Sierra
