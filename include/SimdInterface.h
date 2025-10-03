// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SIMDINTERFACE_H
#define SIMDINTERFACE_H

/** \file
 *  \brief STK SIMD Interface
 *
 *  Nalu wrapper to provide SIMD capability for vectorizing sierra::nalu::Kernel
 *  algorithms.
 */

#include "stk_simd/Simd.hpp"
#include "Kokkos_Macros.hpp"

#include <vector>

namespace sierra {
namespace nalu {

typedef stk::simd::Double SimdDouble;

typedef SimdDouble DoubleType;

template <typename T>
using AlignedVector = std::vector<T>;

using ScalarAlignedVector = AlignedVector<DoubleType>;

static constexpr int simdLen = stk::simd::ndoubles;

KOKKOS_INLINE_FUNCTION
size_t
get_num_simd_groups(size_t length)
{
  size_t numSimdGroups = length / simdLen;
  const size_t remainder = length % simdLen;
  if (remainder > 0) {
    numSimdGroups += 1;
  }
  return numSimdGroups;
}

KOKKOS_INLINE_FUNCTION
int
get_length_of_next_simd_group(int index, int length)
{
  int nextLength = simdLen;
  if (length - index * simdLen < simdLen) {
    nextLength = length - index * simdLen;
  }
  if (nextLength < 0 || nextLength > simdLen) {
    nextLength = 0;
  }
  return nextLength;
}

} // namespace nalu
} // namespace sierra

typedef sierra::nalu::DoubleType DoubleType;
#endif /* SIMDINTERFACE_H */
