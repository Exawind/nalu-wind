// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef KOKKOS_FRAMEWORK_H
#define KOKKOS_FRAMEWORK_H

#include "Kokkos_Core.hpp"

#include "stk_simd/Simd.hpp"

#include <type_traits>

namespace sierra {
namespace nalu {
namespace matrix_free {

using exec_space = Kokkos::DefaultExecutionSpace;

#ifndef USE_STK_SIMD_NONE
template <typename ExecSpace>
struct ExecTraits
{
  using data_type = stk::simd::Double;
  using memory_traits = Kokkos::MemoryTraits<Kokkos::Restrict>;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutRight;
  static constexpr int alignment = alignof(data_type);
  static constexpr int simd_len = stk::simd::ndoubles;
};
#else
template <typename ExecSpace>
struct ExecTraits
{
  using data_type = double;
  using memory_traits = Kokkos::MemoryTraits<Kokkos::Restrict>;
  using memory_space = typename ExecSpace::memory_space;
  using layout = Kokkos::LayoutRight;
  static constexpr int alignment = alignof(data_type);
  static constexpr int simd_len = 1;
};
#endif

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct ExecTraits<Kokkos::Cuda>
{
  using data_type = double;
  using memory_traits =
    Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Aligned>;
  using memory_space = typename Kokkos::Cuda::memory_space;
  using layout = Kokkos::LayoutLeft;
  static constexpr int alignment = alignof(data_type);
  static constexpr int simd_len = 1;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
struct ExecTraits<Kokkos::Experimental::HIP>
{
  using data_type = double;
  using memory_traits =
    Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Aligned>;
  using memory_space = typename Kokkos::Experimental::HIP::memory_space;
  using layout = Kokkos::LayoutLeft;
  static constexpr int alignment = alignof(data_type);
  static constexpr int simd_len = 1;
};
#endif

using ftype = typename ExecTraits<exec_space>::data_type;
static constexpr int simd_len = ExecTraits<exec_space>::simd_len;
static constexpr int alignment = ExecTraits<exec_space>::alignment;

} // namespace matrix_free
} // namespace nalu
} // namespace sierra

#endif
