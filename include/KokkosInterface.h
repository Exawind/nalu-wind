// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef INCLUDE_KOKKOSINTERFACE_H_
#define INCLUDE_KOKKOSINTERFACE_H_

#include <cstddef>

#include <stk_mesh/base/Entity.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define NALU_ALIGNED alignas(sizeof(double))
#elif defined(NALU_USE_POWER9_ALIGNMENT)
#define NALU_ALIGNED alignas(16)
#else
#define NALU_ALIGNED alignas(KOKKOS_MEMORY_ALIGNMENT)
#endif

#if defined(__INTEL_COMPILER)
#define POINTER_RESTRICT restrict
#elif defined(__GNUC__)
#define POINTER_RESTRICT __restrict__
#else
#define POINTER_RESTRICT
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define NONCONST_LAMBDA [&] __host__
#define KOKKOS_ENABLE_GPU
#else
#define NONCONST_LAMBDA [&]
#undef KOKKOS_ENABLE_GPU
#endif

namespace sierra {
namespace nalu {

#if defined(KOKKOS_ENABLE_CUDA)
typedef Kokkos::CudaSpace MemSpace;
#elif defined(KOKKOS_ENABLE_HIP)
typedef Kokkos::Experimental::HIPSpace MemSpace;
#elif defined(KOKKOS_HAVE_OPENMP)
typedef Kokkos::OpenMP MemSpace;
#else
typedef Kokkos::HostSpace MemSpace;
#endif

#if defined(KOKKOS_ENABLE_HIP)
#define NTHREADS_PER_DEVICE_TEAM 128
#else
#define NTHREADS_PER_DEVICE_TEAM Kokkos::AUTO
#endif

using HostSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceSpace = Kokkos::DefaultExecutionSpace;

using LinSysMemSpace = DeviceSpace::memory_space;

using DeviceShmem = DeviceSpace::scratch_memory_space;
using HostShmem = HostSpace::scratch_memory_space;
using DynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
using TeamHandleType =
  Kokkos::TeamPolicy<HostSpace, DynamicScheduleType>::member_type;

#if defined(KOKKOS_ENABLE_HIP)
using DeviceTeamHandleType = Kokkos::TeamPolicy<
  DeviceSpace,
  Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>,
  DynamicScheduleType>::member_type;
using DeviceRangePolicy = Kokkos::
  RangePolicy<DeviceSpace, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>>;
using DeviceTeamPolicy = Kokkos::
  TeamPolicy<DeviceSpace, Kokkos::LaunchBounds<NTHREADS_PER_DEVICE_TEAM, 1>>;
#else
using DeviceTeamHandleType =
  Kokkos::TeamPolicy<DeviceSpace, DynamicScheduleType>::member_type;
using DeviceRangePolicy = Kokkos::RangePolicy<DeviceSpace>;
using DeviceTeamPolicy = Kokkos::TeamPolicy<DeviceSpace>;
#endif

template <typename T, typename SHMEM = HostShmem>
using SharedMemView =
  Kokkos::View<T, Kokkos::LayoutRight, SHMEM, Kokkos::MemoryUnmanaged>;

#if defined(KOKKOS_ENABLE_HIP)
template <typename T>
using AlignedViewType = Kokkos::View<T>;
#else
template <typename T>
using AlignedViewType = Kokkos::View<T, Kokkos::MemoryTraits<Kokkos::Aligned>>;
#endif

using HostTeamPolicy = Kokkos::TeamPolicy<HostSpace>;
using HostRangePolicy = Kokkos::RangePolicy<HostSpace>;
using DeviceTeam = DeviceTeamPolicy::member_type;
using HostTeam = HostTeamPolicy::member_type;

inline HostTeamPolicy
get_host_team_policy(
  const size_t sz, const size_t bytes_per_team, const size_t bytes_per_thread)
{
  HostTeamPolicy policy(sz, Kokkos::AUTO);
  return policy.set_scratch_size(
    1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

inline DeviceTeamPolicy
get_device_team_policy(
  const size_t sz,
  const size_t bytes_per_team,
  const size_t bytes_per_thread,
  const int threads_per_team)
{
  DeviceTeamPolicy policy(sz, threads_per_team);
  return policy.set_scratch_size(
    1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

inline DeviceTeamPolicy
get_device_team_policy(
  const size_t sz, const size_t bytes_per_team, const size_t bytes_per_thread)
{
  DeviceTeamPolicy policy(sz, NTHREADS_PER_DEVICE_TEAM);
  return policy.set_scratch_size(
    1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

template <
  typename T,
  typename TEAMHANDLETYPE,
  typename TeamShmemType = HostShmem>
KOKKOS_FUNCTION SharedMemView<T*, TeamShmemType>
get_shmem_view_1D(const TEAMHANDLETYPE& team, size_t len)
{
  return Kokkos::subview(
    SharedMemView<T**, TeamShmemType>(
      team.team_scratch(1), team.team_size(), len),
    team.team_rank(), Kokkos::ALL());
}

template <
  typename T,
  typename TEAMHANDLETYPE,
  typename TeamShmemType = HostShmem>
KOKKOS_FUNCTION SharedMemView<T**, TeamShmemType>
get_shmem_view_2D(const TEAMHANDLETYPE& team, size_t len1, size_t len2)
{
  return Kokkos::subview(
    SharedMemView<T***, TeamShmemType>(
      team.team_scratch(1), team.team_size(), len1, len2),
    team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
}

template <
  typename T,
  typename TEAMHANDLETYPE,
  typename TeamShmemType = HostShmem>
KOKKOS_FUNCTION SharedMemView<T***, TeamShmemType>
get_shmem_view_3D(
  const TEAMHANDLETYPE& team, size_t len1, size_t len2, size_t len3)
{
  return Kokkos::subview(
    SharedMemView<T****, TeamShmemType>(
      team.team_scratch(1), team.team_size(), len1, len2, len3),
    team.team_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
}

#if !defined(KOKKOS_ENABLE_GPU)
template <
  typename T,
  typename TEAMHANDLETYPE,
  typename TeamShmemType = HostShmem>
KOKKOS_FUNCTION SharedMemView<T****, TeamShmemType>
get_shmem_view_4D(
  const TEAMHANDLETYPE& team,
  size_t len1,
  size_t len2,
  size_t len3,
  size_t len4)
{
  return Kokkos::subview(
    SharedMemView<T*****, TeamShmemType>(
      team.team_scratch(1), team.team_size(), len1, len2, len3, len4),
    team.team_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
}

template <
  typename T,
  typename TEAMHANDLETYPE,
  typename TeamShmemType = HostShmem>
KOKKOS_FUNCTION SharedMemView<T*****, TeamShmemType>
get_shmem_view_5D(
  const TEAMHANDLETYPE& team,
  size_t len1,
  size_t len2,
  size_t len3,
  size_t len4,
  size_t len5)
{
  return Kokkos::subview(
    SharedMemView<T******, TeamShmemType>(
      team.team_scratch(1), team.team_size(), len1, len2, len3, len4, len5),
    team.team_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
}
#endif

template <typename SizeType, class Function>
void
kokkos_parallel_for(
  const std::string& debuggingName, SizeType n, Function loop_body)
{
  Kokkos::parallel_for(
    debuggingName, Kokkos::RangePolicy<Kokkos::Serial>(0, n), loop_body);
}

template <typename SizeType, class Function, typename ReduceType>
void
kokkos_parallel_reduce(
  SizeType n,
  Function loop_body,
  ReduceType& reduce,
  const std::string& debuggingName)
{
  Kokkos::parallel_reduce(
    debuggingName, Kokkos::RangePolicy<Kokkos::Serial>(0, n), loop_body,
    reduce);
}

template <typename T, typename MemorySpace = MemSpace>
inline T*
kokkos_malloc_on_device(const std::string& debuggingName)
{
  return static_cast<T*>(
    Kokkos::kokkos_malloc<MemorySpace>(debuggingName, sizeof(T)));
}

template <typename MemorySpace = MemSpace>
inline void
kokkos_free_on_device(void* ptr)
{
  Kokkos::kokkos_free<MemorySpace>(ptr);
}

template <typename ViewType, typename T>
KOKKOS_FUNCTION void
set_vals(ViewType& view, const T& val)
{
  unsigned length = view.size();
  auto* viewData = view.data();
  for (unsigned i = 0; i < length; ++i) {
    viewData[i] = val;
  }
}

} // namespace nalu
} // namespace sierra

#endif /* INCLUDE_KOKKOSINTERFACE_H_ */
