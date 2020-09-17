// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include "matrix_free/LocalArray.h"

#include "Kokkos_Macros.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

#define XH 0
#define YH 1
#define ZH 2

template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION Scalar
determinant(const LocalArray<Scalar[3][3]>& jac)
{
  return jac(XH, XH) * (jac(YH, YH) * jac(ZH, ZH) - jac(YH, ZH) * jac(ZH, YH)) -
         jac(XH, YH) * (jac(YH, XH) * jac(ZH, ZH) - jac(YH, ZH) * jac(ZH, XH)) +
         jac(XH, ZH) * (jac(YH, XH) * jac(ZH, YH) - jac(YH, YH) * jac(ZH, XH));
}

template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION void
adjugate_matrix(
  const LocalArray<Scalar[3][3]>& mat, LocalArray<Scalar[3][3]>& adj)
{
  adj(XH, XH) = mat(YH, YH) * mat(ZH, ZH) - mat(ZH, YH) * mat(YH, ZH);
  adj(XH, YH) = mat(YH, ZH) * mat(ZH, XH) - mat(ZH, ZH) * mat(YH, XH);
  adj(XH, ZH) = mat(YH, XH) * mat(ZH, YH) - mat(ZH, XH) * mat(YH, YH);
  adj(YH, XH) = mat(XH, ZH) * mat(ZH, YH) - mat(ZH, ZH) * mat(XH, YH);
  adj(YH, YH) = mat(XH, XH) * mat(ZH, ZH) - mat(ZH, XH) * mat(XH, ZH);
  adj(YH, ZH) = mat(XH, YH) * mat(ZH, XH) - mat(ZH, YH) * mat(XH, XH);
  adj(ZH, XH) = mat(XH, YH) * mat(YH, ZH) - mat(YH, YH) * mat(XH, ZH);
  adj(ZH, YH) = mat(XH, ZH) * mat(YH, XH) - mat(YH, ZH) * mat(XH, XH);
  adj(ZH, ZH) = mat(XH, XH) * mat(YH, YH) - mat(YH, XH) * mat(XH, YH);
}

template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION LocalArray<Scalar[3][3]>
invert_matrix(const LocalArray<Scalar[3][3]>& mat)
{
  LocalArray<Scalar[3][3]> inv;
  auto inv_detj = 1. / determinant(mat);
  inv(XH, XH) =
    inv_detj * (mat(YH, YH) * mat(ZH, ZH) - mat(ZH, YH) * mat(YH, ZH));
  inv(XH, YH) =
    inv_detj * (mat(YH, ZH) * mat(ZH, XH) - mat(ZH, ZH) * mat(YH, XH));
  inv(XH, ZH) =
    inv_detj * (mat(YH, XH) * mat(ZH, YH) - mat(ZH, XH) * mat(YH, YH));
  inv(YH, XH) =
    inv_detj * (mat(XH, ZH) * mat(ZH, YH) - mat(ZH, ZH) * mat(XH, YH));
  inv(YH, YH) =
    inv_detj * (mat(XH, XH) * mat(ZH, ZH) - mat(ZH, XH) * mat(XH, ZH));
  inv(YH, ZH) =
    inv_detj * (mat(XH, YH) * mat(ZH, XH) - mat(ZH, YH) * mat(XH, XH));
  inv(ZH, XH) =
    inv_detj * (mat(XH, YH) * mat(YH, ZH) - mat(YH, YH) * mat(XH, ZH));
  inv(ZH, YH) =
    inv_detj * (mat(XH, ZH) * mat(YH, XH) - mat(YH, ZH) * mat(XH, XH));
  inv(ZH, ZH) =
    inv_detj * (mat(XH, XH) * mat(YH, YH) - mat(YH, XH) * mat(XH, YH));

  return inv;
}

template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION LocalArray<Scalar[3][3]>
invert_transpose_matrix(const LocalArray<Scalar[3][3]>& mat)
{
  LocalArray<Scalar[3][3]> inv;
  auto inv_detj = 1. / determinant(mat);
  inv(XH, XH) =
    inv_detj * (mat(YH, YH) * mat(ZH, ZH) - mat(ZH, YH) * mat(YH, ZH));
  inv(YH, XH) =
    inv_detj * (mat(YH, ZH) * mat(ZH, XH) - mat(ZH, ZH) * mat(YH, XH));
  inv(ZH, XH) =
    inv_detj * (mat(YH, XH) * mat(ZH, YH) - mat(ZH, XH) * mat(YH, YH));
  inv(XH, YH) =
    inv_detj * (mat(XH, ZH) * mat(ZH, YH) - mat(ZH, ZH) * mat(XH, YH));
  inv(YH, YH) =
    inv_detj * (mat(XH, XH) * mat(ZH, ZH) - mat(ZH, XH) * mat(XH, ZH));
  inv(ZH, YH) =
    inv_detj * (mat(XH, YH) * mat(ZH, XH) - mat(ZH, YH) * mat(XH, XH));
  inv(XH, ZH) =
    inv_detj * (mat(XH, YH) * mat(YH, ZH) - mat(YH, YH) * mat(XH, ZH));
  inv(YH, ZH) =
    inv_detj * (mat(XH, ZH) * mat(YH, XH) - mat(YH, ZH) * mat(XH, XH));
  inv(ZH, ZH) =
    inv_detj * (mat(XH, XH) * mat(YH, YH) - mat(YH, XH) * mat(XH, YH));

  return inv;
}

template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION LocalArray<Scalar[3][3]>
adjugate_matrix(const LocalArray<Scalar[3][3]>& mat)
{
  return LocalArray<Scalar[3][3]>{
    {{mat(YH, YH) * mat(ZH, ZH) - mat(ZH, YH) * mat(YH, ZH),
      mat(YH, ZH) * mat(ZH, XH) - mat(ZH, ZH) * mat(YH, XH),
      mat(YH, XH) * mat(ZH, YH) - mat(ZH, XH) * mat(YH, YH)},
     {mat(XH, ZH) * mat(ZH, YH) - mat(ZH, ZH) * mat(XH, YH),
      mat(XH, XH) * mat(ZH, ZH) - mat(ZH, XH) * mat(XH, ZH),
      mat(XH, YH) * mat(ZH, XH) - mat(ZH, YH) * mat(XH, XH)},
     {mat(XH, YH) * mat(YH, ZH) - mat(YH, YH) * mat(XH, ZH),
      mat(XH, ZH) * mat(YH, XH) - mat(YH, ZH) * mat(XH, XH),
      mat(XH, XH) * mat(YH, YH) - mat(YH, XH) * mat(XH, YH)}}};
}

template <typename InpScalar, typename OutScalar>
KOKKOS_FORCEINLINE_FUNCTION void
transform_vector(
  const LocalArray<InpScalar[3][3]>& A,
  const Kokkos::Array<OutScalar, 3>& x,
  Kokkos::Array<OutScalar, 3>& y)
{
  y[XH] = A(XH, XH) * x[XH] + A(XH, YH) * x[YH] + A(XH, ZH) * x[ZH];
  y[YH] = A(YH, XH) * x[XH] + A(YH, YH) * x[YH] + A(YH, ZH) * x[ZH];
  y[ZH] = A(ZH, XH) * x[XH] + A(ZH, YH) * x[YH] + A(ZH, ZH) * x[ZH];
}

template <typename InpScalar, typename OutScalar>
KOKKOS_FORCEINLINE_FUNCTION void
transform_tensor(
  const LocalArray<InpScalar[3][3]>& A,
  const LocalArray<OutScalar[3][3]>& x,
  LocalArray<OutScalar[3][3]>& y)
{
  for (int d = 0; d < 3; ++d) {
    y(d, XH) =
      A(XH, XH) * x(d, XH) + A(XH, YH) * x(d, YH) + A(XH, ZH) * x(d, ZH);
    y(d, YH) =
      A(YH, XH) * x(d, XH) + A(YH, YH) * x(d, YH) + A(YH, ZH) * x(d, ZH);
    y(d, ZH) =
      A(ZH, XH) * x(d, XH) + A(ZH, YH) * x(d, YH) + A(ZH, ZH) * x(d, ZH);
  }
} // namespace matrix_free

#undef XH
#undef YH
#undef ZH

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
