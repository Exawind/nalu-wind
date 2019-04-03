/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/
#ifndef TensorOps_h
#define TensorOps_h

#include <array>
#include <limits>

#include <stk_util/util/ReportHandler.hpp>

#include <KokkosInterface.h>
#include <SimdInterface.h>

namespace sierra{
namespace nalu{

  KOKKOS_INLINE_FUNCTION constexpr double tiny_positive_value() {
    return (1.0e6*std::numeric_limits<double>::min());
  }

  KOKKOS_INLINE_FUNCTION constexpr double small_positive_value() {
    return 1.0e2*std::numeric_limits<double>::epsilon();
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION ScalarType vecnorm_sq2(const ScalarType* x) {
    return (x[0] * x[0] + x[1] * x[1]);
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION ScalarType vecnorm_sq3(const ScalarType* x) {
    return (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  }

  template <typename ScalarType>
  void normalize_vec3(ScalarType* x) {
    const ScalarType invmag = 1.0 / stk::math::sqrt(vecnorm_sq3(x));
    x[0] *= invmag;
    x[1] *= invmag;
    x[2] *= invmag;
  }

  template <typename ScalarType>
  ScalarType ddot(const ScalarType* u, const ScalarType* v, int n)
  {
    ScalarType val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += u[i] * v[i];
    }
    return val;
  }

  template <typename ScalarType>
  void cross3(const ScalarType* u, const ScalarType* v, ScalarType* cross)
  {
    cross[0] =   u[1]*v[2] - u[2]*v[1];
    cross[1] = -(u[0]*v[2] - u[2]*v[0]);
    cross[2] =   u[0]*v[1] - u[1]*v[0];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION ScalarType determinant22(const ScalarType* mat)
  {
     enum { XX = 0, XY = 1, YX = 2, YY = 3 };
     return (mat[XX] * mat[YY] - mat[XY] * mat[YX]);
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION ScalarType determinant33(const ScalarType* mat)  {
    enum { XX = 0, XY = 1, XZ = 2, YX = 3, YY = 4, YZ = 5, ZX = 6, ZY = 7, ZZ = 8 };
    return (mat[XX] * (mat[YY] * mat[ZZ] - mat[YZ] * mat[ZY])
          - mat[XY] * (mat[YX] * mat[ZZ] - mat[YZ] * mat[ZX])
          + mat[XZ] * (mat[YX] * mat[ZY] - mat[YY] * mat[ZX]));
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void adjugate_matrix33(const ScalarType jact[3][3], ScalarType adjJac[3][3])
  {
    adjJac[0][0] = jact[1][1] * jact[2][2] - jact[2][1] * jact[1][2];
    adjJac[0][1] = jact[1][2] * jact[2][0] - jact[2][2] * jact[1][0];
    adjJac[0][2] = jact[1][0] * jact[2][1] - jact[2][0] * jact[1][1];

    adjJac[1][0] = jact[0][2] * jact[2][1] - jact[2][2] * jact[0][1];
    adjJac[1][1] = jact[0][0] * jact[2][2] - jact[2][0] * jact[0][2];
    adjJac[1][2] = jact[0][1] * jact[2][0] - jact[2][1] * jact[0][0];

    adjJac[2][0] = jact[0][1] * jact[1][2] - jact[1][1] * jact[0][2];
    adjJac[2][1] = jact[0][2] * jact[1][0] - jact[1][2] * jact[0][0];
    adjJac[2][2] = jact[0][0] * jact[1][1] - jact[1][0] * jact[0][1];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void invert_matrix33(const ScalarType A[3][3], ScalarType Ainv[3][3])
  {
    ScalarType inv_detj = 1.0 / determinant33(&A[0][0]);

    Ainv[0][0] = inv_detj*(A[1][1] * A[2][2] - A[2][1] * A[1][2]);
    Ainv[0][1] = inv_detj*(A[1][2] * A[2][0] - A[2][2] * A[1][0]);
    Ainv[0][2] = inv_detj*(A[1][0] * A[2][1] - A[2][0] * A[1][1]);

    Ainv[1][0] = inv_detj*(A[0][2] * A[2][1] - A[2][2] * A[0][1]);
    Ainv[1][1] = inv_detj*(A[0][0] * A[2][2] - A[2][0] * A[0][2]);
    Ainv[1][2] = inv_detj*(A[0][1] * A[2][0] - A[2][1] * A[0][0]);

    Ainv[2][0] = inv_detj*(A[0][1] * A[1][2] - A[1][1] * A[0][2]);
    Ainv[2][1] = inv_detj*(A[0][2] * A[1][0] - A[1][2] * A[0][0]);
    Ainv[2][2] = inv_detj*(A[0][0] * A[1][1] - A[1][0] * A[0][1]);
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void invert_matrix33(
    const ScalarType* POINTER_RESTRICT A,
    ScalarType* POINTER_RESTRICT Ainv)
  {
    enum { XX = 0, XY = 1, XZ = 2, YX = 3, YY = 4, YZ = 5, ZX = 6, ZY = 7, ZZ = 8 };
    ScalarType inv_detj = 1.0 / determinant33(A);

    Ainv[XX] = inv_detj*(A[YY] * A[ZZ] - A[ZY] * A[YZ]);
    Ainv[XY] = inv_detj*(A[YZ] * A[ZX] - A[ZZ] * A[YX]);
    Ainv[XZ] = inv_detj*(A[YX] * A[ZY] - A[ZX] * A[YY]);
    Ainv[YX] = inv_detj*(A[XZ] * A[ZY] - A[ZZ] * A[XY]);
    Ainv[YY] = inv_detj*(A[XX] * A[ZZ] - A[ZX] * A[XZ]);
    Ainv[YZ] = inv_detj*(A[XY] * A[ZX] - A[ZY] * A[XX]);
    Ainv[ZX] = inv_detj*(A[XY] * A[YZ] - A[YY] * A[XZ]);
    Ainv[ZY] = inv_detj*(A[XZ] * A[YZ] - A[YZ] * A[XX]);
    Ainv[ZZ] = inv_detj*(A[XX] * A[YY] - A[YX] * A[XY]);
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void solve22(
    const ScalarType* POINTER_RESTRICT A,
    const ScalarType* POINTER_RESTRICT b,
    ScalarType*  POINTER_RESTRICT x)
  {
    ThrowAssert(stk::simd::are_any(determinant22(A) > tiny_positive_value()));
    enum { XX = 0, XY = 1, YX = 2, YY = 3 };
    enum { X_RANK1 = 0, Y_RANK1 = 1};

    const ScalarType inv_detA = 1.0 / determinant22(A);
    x[X_RANK1] =  (A[YY] * b[X_RANK1] - A[XY] * b[Y_RANK1]) * inv_detA;
    x[Y_RANK1] = -(A[YX] * b[X_RANK1] - A[XX] * b[Y_RANK1]) * inv_detA;
  }

  // computes b = A^-1 x;
  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void solve33(
    const ScalarType* POINTER_RESTRICT A,
    const ScalarType* POINTER_RESTRICT b,
    ScalarType*  POINTER_RESTRICT x)
  {
    ThrowAssert(stk::simd::are_any(determinant33(A) > tiny_positive_value()));


    enum { XX = 0, XY = 1, XZ = 2, YX = 3, YY = 4, YZ = 5, ZX = 6, ZY = 7, ZZ = 8 };
    enum { X_RANK1 = 0, Y_RANK1 = 1, Z_RANK1 = 2 };

    const ScalarType inv_detA = 1.0 / determinant33(A);
    x[X_RANK1] = ((A[YY] * A[ZZ] - A[YZ] * A[ZY]) * b[X_RANK1] + (A[XZ] * A[ZY] - A[XY] * A[ZZ]) * b[Y_RANK1] +
                  (A[XY] * A[YZ] - A[XZ] * A[YY]) * b[Z_RANK1]) *
                 inv_detA;

    x[Y_RANK1] = ((A[YZ] * A[ZX] - A[YX] * A[ZZ]) * b[X_RANK1] + (A[XX] * A[ZZ] - A[XZ] * A[ZX]) * b[Y_RANK1] +
                  (A[XZ] * A[YX] - A[XX] * A[YZ]) * b[Z_RANK1]) *
                 inv_detA;

    x[Z_RANK1] = ((A[YX] * A[ZY] - A[YY] * A[ZX]) * b[X_RANK1] + (A[XY] * A[ZX] - A[XX] * A[ZY]) * b[Y_RANK1] +
                  (A[XX] * A[YY] - A[XY] * A[YX]) * b[Z_RANK1]) *
                 inv_detA;
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void matvec22(const ScalarType* A, const ScalarType* x, ScalarType* b)
  {
    b[0] = A[0] * x[0] + A[1] * x[1];
    b[1] = A[2] * x[0] + A[3] * x[1];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void mxm22(const ScalarType* A, const ScalarType* B, ScalarType* C)
  {
    C[0] = A[0] * B[0] + A[1] * B[2];
    C[1] = A[0] * B[1] + A[1] * B[3];
    C[2] = A[2] * B[0] + A[3] * B[2];
    C[3] = A[2] * B[1] + A[3] * B[3];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void mxm33(const ScalarType* A, const ScalarType* B, ScalarType* C)
  {
    enum { XX = 0, XY = 1, XZ = 2, YX = 3, YY = 4, YZ = 5, ZX = 6, ZY = 7, ZZ = 8 };
    C[XX] = A[XX] * B[XX] + A[XY] * B[YX] + A[XZ] * B[ZX];
    C[XY] = A[XX] * B[XY] + A[XY] * B[YY] + A[XZ] * B[ZY];
    C[XZ] = A[XX] * B[XZ] + A[XY] * B[YZ] + A[XZ] * B[ZZ];
    C[YX] = A[YX] * B[XX] + A[YY] * B[YX] + A[YZ] * B[ZX];
    C[YY] = A[YX] * B[XY] + A[YY] * B[YY] + A[YZ] * B[ZY];
    C[YZ] = A[YX] * B[XZ] + A[YY] * B[YZ] + A[YZ] * B[ZZ];
    C[ZX] = A[ZX] * B[XX] + A[ZY] * B[YX] + A[ZZ] * B[ZX];
    C[ZY] = A[ZX] * B[XY] + A[ZY] * B[YY] + A[ZZ] * B[ZY];
    C[ZZ] = A[ZX] * B[XZ] + A[ZY] * B[YZ] + A[ZZ] * B[ZZ];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void matvec33(const ScalarType* A, const ScalarType* x, ScalarType* b)
  {
    b[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    b[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    b[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void transpose22(const ScalarType* A, ScalarType* At)
  {
    At[0] = A[0];
    At[1] = A[2];
    At[2] = A[1];
    At[3] = A[3];
  }

  template <typename ScalarType>
  KOKKOS_FORCEINLINE_FUNCTION void transpose33(const ScalarType* A, ScalarType* At)
  {
    enum { XX = 0, XY = 1, XZ = 2, YX = 3, YY = 4, YZ = 5, ZX = 6, ZY = 7, ZZ = 8 };
    At[XX] = A[XX];
    At[YY] = A[YY];
    At[ZZ] = A[ZZ];

    At[XY] = A[YX];
    At[XZ] = A[ZX];
    At[YX] = A[XY];
    At[YZ] = A[ZY];
    At[ZX] = A[XZ];
    At[ZY] = A[YZ];
  }

}
}

#endif
