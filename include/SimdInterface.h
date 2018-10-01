/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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

template<typename T>
using AlignedVector = std::vector<T, non_std::AlignedAllocator<T, KOKKOS_MEMORY_ALIGNMENT>>;

using ScalarAlignedVector = AlignedVector<DoubleType>;

static constexpr int simdLen = stk::simd::ndoubles;

inline
size_t get_num_simd_groups(size_t length)
{
    size_t numSimdGroups = length/simdLen;
    const size_t remainder = length%simdLen;
    if (remainder > 0) {
      numSimdGroups += 1;
    }
    return numSimdGroups;
}

inline
int get_length_of_next_simd_group(int index, int length)
{
  int nextLength = simdLen;
  if (length - index*simdLen < simdLen) {
    nextLength = length - index*simdLen;
  }
  if (nextLength < 0 || nextLength > simdLen) {
    nextLength = 0;
  }
  return nextLength;
}

}  // nalu
}  // sierra

typedef sierra::nalu::DoubleType DoubleType;
#endif /* SIMDINTERFACE_H */
