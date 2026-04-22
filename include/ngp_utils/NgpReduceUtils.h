// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NGPREDUCEUTILS_H
#define NGPREDUCEUTILS_H

#include <cfloat>

#include "KokkosInterface.h"
#include "SimdInterface.h"

namespace sierra {
namespace kynema_ugf {
namespace kynema_ugf_ngp {

/** A custom Kokkos reduction operator for array types
 *
 *  Useful when you want to accumulate multiple quantities, e.g., computing an
 *  area-weighted average.
 */
template <typename ScalarType, int N>
struct NgpReduceArray
{
  ScalarType array_[N];

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray() {}

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray(ScalarType val)
  {
    for (int i = 0; i < N; ++i)
      array_[i] = val;
  }

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray(const NgpReduceArray& rhs)
  {
    for (int i = 0; i < N; ++i)
      array_[i] = rhs.array_[i];
  }

  // See discussion in https://github.com/trilinos/Trilinos/issues/6125 for
  // details on the overloads.

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray& operator=(const NgpReduceArray& rhs)
  {
    for (int i = 0; i < N; ++i)
      array_[i] = rhs.array_[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray& operator=(const volatile NgpReduceArray& rhs)
  {
    for (int i = 0; i < N; ++i)
      array_[i] = rhs.array_[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile NgpReduceArray& operator=(const NgpReduceArray& rhs) volatile
  {
    for (int i = 0; i < N; ++i)
      array_[i] = rhs.array_[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  volatile NgpReduceArray&
  operator=(const volatile NgpReduceArray& rhs) volatile
  {
    for (int i = 0; i < N; ++i)
      array_[i] = rhs.array_[i];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const NgpReduceArray& rhs)
  {
    for (int i = 0; i < N; ++i)
      array_[i] += rhs.array_[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile NgpReduceArray& rhs) volatile
  {
    for (int i = 0; i < N; ++i)
      array_[i] += rhs.array_[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator*=(const NgpReduceArray& rhs)
  {
    for (int i = 0; i < N; ++i)
      array_[i] *= rhs.array_[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator*=(const volatile NgpReduceArray& rhs) volatile
  {
    for (int i = 0; i < N; ++i)
      array_[i] *= rhs.array_[i];
  }
};

using ArrayDbl2 = NgpReduceArray<double, 2>;
using ArrayDbl3 = NgpReduceArray<double, 3>;
using ArrayInt2 = NgpReduceArray<int, 2>;

using ArraySimdDouble2 = NgpReduceArray<DoubleType, 2>;
using ArraySimdDouble3 = NgpReduceArray<DoubleType, 3>;

/** Utility function for reduction accumulation
 *
 *  This function is necessary when looping over elements in bucket+SIMD loops
 *  where we might not have enough elements (i.e., numSimdElems < simdLen), we
 *  don't want to accumulate bad data.
 */
KOKKOS_INLINE_FUNCTION
void
simd_reduce_sum(double& out, const DoubleType& inp, int len)
{
  for (int i = 0; i < len; ++i)
    out += stk::simd::get_data(inp, i);
}

} // namespace kynema_ugf_ngp
} // namespace kynema_ugf
} // namespace sierra

namespace Kokkos {

template <>
struct reduction_identity<sierra::kynema_ugf::DoubleType>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::DoubleType sum() { return DoubleType(0.0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::DoubleType prod() { return DoubleType(1.0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::DoubleType max() { return DoubleType(-DBL_MAX); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::DoubleType min() { return DoubleType(DBL_MAX); }
};

template <>
struct reduction_identity<sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl2>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl2 sum()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl2(0.0);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl2 prod()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl2(1.0);
  }
};

template <>
struct reduction_identity<sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl3>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl3 sum()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl3(0.0);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl3 prod()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArrayDbl3(1.0);
  }
};

template <>
struct reduction_identity<sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble2>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble2 sum()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble2(
      DoubleType(0.0));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble2 prod()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble2(
      DoubleType(1.0));
  }
};

template <>
struct reduction_identity<sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble3>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble3 sum()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble3(
      DoubleType(0.0));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble3 prod()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArraySimdDouble3(
      DoubleType(1.0));
  }
};

template <>
struct reduction_identity<sierra::kynema_ugf::kynema_ugf_ngp::ArrayInt2>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArrayInt2 sum()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArrayInt2(0);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::kynema_ugf::kynema_ugf_ngp::ArrayInt2 prod()
  {
    return sierra::kynema_ugf::kynema_ugf_ngp::ArrayInt2(1);
  }
};

} // namespace Kokkos

#endif /* NGPREDUCEUTILS_H */
