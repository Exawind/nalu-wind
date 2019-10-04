/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef NGPREDUCEUTILS_H
#define NGPREDUCEUTILS_H

#include <cfloat>

#include "KokkosInterface.h"
#include "SimdInterface.h"

namespace sierra {
namespace nalu {
namespace nalu_ngp {

/** A custom Kokkos reduction operator for array types
 *
 *  Useful when you want to accumulate multiple quantities, e.g., computing an
 *  area-weighted average.
 */
template<typename ScalarType, int N>
struct NgpReduceArray
{
  ScalarType array_[N];

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray()
  {}

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray(ScalarType val)
  {
    for (int i=0; i < N; ++i)
      array_[i] = val;
  }

  KOKKOS_INLINE_FUNCTION
  NgpReduceArray(const NgpReduceArray& rhs)
  {
    for (int i=0; i < N; ++i)
      array_[i] = rhs.array_[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const NgpReduceArray& rhs)
  {
    for (int i=0; i < N; ++i)
      array_[i] += rhs.array_[i];
  }

  KOKKOS_INLINE_FUNCTION
  void operator*=(const NgpReduceArray& rhs)
  {
    for (int i=0; i < N; ++i)
      array_[i] *= rhs.array_[i];
  }
};

using ArrayDbl2 = NgpReduceArray<double, 2>;
using ArrayDbl3 = NgpReduceArray<double, 3>;
using ArrayInt2 = NgpReduceArray<int, 2>;

using ArraySimdDouble2 = NgpReduceArray<DoubleType, 2>;
using ArraySimdDouble3 = NgpReduceArray<DoubleType, 3>;

}  // nalu_ngp
}  // nalu
}  // sierra

namespace Kokkos {

template<>
struct reduction_identity<sierra::nalu::DoubleType>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::DoubleType sum()
  { return DoubleType(0.0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::DoubleType prod()
  { return DoubleType(1.0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::DoubleType max()
  { return DoubleType(-DBL_MAX); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::DoubleType min()
  { return DoubleType(DBL_MAX); }
};

template<>
struct reduction_identity<sierra::nalu::nalu_ngp::ArrayDbl2>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArrayDbl2 sum()
  { return sierra::nalu::nalu_ngp::ArrayDbl2(0.0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArrayDbl2 prod()
  { return sierra::nalu::nalu_ngp::ArrayDbl2(1.0); }
};

template<>
struct reduction_identity<sierra::nalu::nalu_ngp::ArrayDbl3>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArrayDbl3 sum()
  { return sierra::nalu::nalu_ngp::ArrayDbl3(0.0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArrayDbl3 prod()
  { return sierra::nalu::nalu_ngp::ArrayDbl3(1.0); }
};

template<>
struct reduction_identity<sierra::nalu::nalu_ngp::ArraySimdDouble2>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArraySimdDouble2 sum()
  { return sierra::nalu::nalu_ngp::ArraySimdDouble2(DoubleType(0.0)); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArraySimdDouble2 prod()
  { return sierra::nalu::nalu_ngp::ArraySimdDouble2(DoubleType(1.0)); }
};

template<>
struct reduction_identity<sierra::nalu::nalu_ngp::ArraySimdDouble3>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArraySimdDouble3 sum()
  { return sierra::nalu::nalu_ngp::ArraySimdDouble3(DoubleType(0.0)); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArraySimdDouble3 prod()
  { return sierra::nalu::nalu_ngp::ArraySimdDouble3(DoubleType(1.0)); }
};

template<>
struct reduction_identity<sierra::nalu::nalu_ngp::ArrayInt2>
{
  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArrayInt2 sum()
  { return sierra::nalu::nalu_ngp::ArrayInt2(0); }

  KOKKOS_FORCEINLINE_FUNCTION
  static sierra::nalu::nalu_ngp::ArrayInt2 prod()
  { return sierra::nalu::nalu_ngp::ArrayInt2(1); }
};

} // namespace Kokkos

#endif /* NGPREDUCEUTILS_H */
