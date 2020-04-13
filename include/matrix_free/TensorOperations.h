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
  const LocalArray<Scalar[3][3]>& jact, LocalArray<Scalar[3][3]>& adjJac)
{
  adjJac(XH, XH) = jact(YH, YH) * jact(ZH, ZH) - jact(ZH, YH) * jact(YH, ZH);
  adjJac(XH, YH) = jact(YH, ZH) * jact(ZH, XH) - jact(ZH, ZH) * jact(YH, XH);
  adjJac(XH, ZH) = jact(YH, XH) * jact(ZH, YH) - jact(ZH, XH) * jact(YH, YH);
  adjJac(YH, XH) = jact(XH, ZH) * jact(ZH, YH) - jact(ZH, ZH) * jact(XH, YH);
  adjJac(YH, YH) = jact(XH, XH) * jact(ZH, ZH) - jact(ZH, XH) * jact(XH, ZH);
  adjJac(YH, ZH) = jact(XH, YH) * jact(ZH, XH) - jact(ZH, YH) * jact(XH, XH);
  adjJac(ZH, XH) = jact(XH, YH) * jact(YH, ZH) - jact(YH, YH) * jact(XH, ZH);
  adjJac(ZH, YH) = jact(XH, ZH) * jact(YH, XH) - jact(YH, ZH) * jact(XH, XH);
  adjJac(ZH, ZH) = jact(XH, XH) * jact(YH, YH) - jact(YH, XH) * jact(XH, YH);
}

template <typename Scalar>
KOKKOS_FORCEINLINE_FUNCTION LocalArray<Scalar[3][3]>
adjugate_matrix(const LocalArray<Scalar[3][3]>& jact)
{
  return LocalArray<Scalar[3][3]>{
    {{jact(YH, YH) * jact(ZH, ZH) - jact(ZH, YH) * jact(YH, ZH),
      jact(YH, ZH) * jact(ZH, XH) - jact(ZH, ZH) * jact(YH, XH),
      jact(YH, XH) * jact(ZH, YH) - jact(ZH, XH) * jact(YH, YH)},
     {jact(XH, ZH) * jact(ZH, YH) - jact(ZH, ZH) * jact(XH, YH),
      jact(XH, XH) * jact(ZH, ZH) - jact(ZH, XH) * jact(XH, ZH),
      jact(XH, YH) * jact(ZH, XH) - jact(ZH, YH) * jact(XH, XH)},
     {jact(XH, YH) * jact(YH, ZH) - jact(YH, YH) * jact(XH, ZH),
      jact(XH, ZH) * jact(YH, XH) - jact(YH, ZH) * jact(XH, XH),
      jact(XH, XH) * jact(YH, YH) - jact(YH, XH) * jact(XH, YH)}}};
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

#undef XH
#undef YH
#undef ZH

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
