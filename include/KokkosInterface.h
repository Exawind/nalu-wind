/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef INCLUDE_KOKKOSINTERFACE_H_
#define INCLUDE_KOKKOSINTERFACE_H_

#include <stk_mesh/base/Entity.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>

#define NALU_ALIGNED alignas(KOKKOS_MEMORY_ALIGNMENT)

#if defined(__INTEL_COMPILER)
#define POINTER_RESTRICT restrict
#elif defined(__GNUC__)
#define POINTER_RESTRICT __restrict__
#else
#define POINTER_RESTRICT
#endif

#ifdef KOKKOS_ENABLE_CUDA
#define NONCONST_LAMBDA [&]__host__
#else
#define NONCONST_LAMBDA [&]
#endif

namespace sierra {
namespace nalu {

#ifdef KOKKOS_ENABLE_CUDA
   typedef Kokkos::CudaUVMSpace::memory_space    MemSpace;
#elif defined(KOKKOS_HAVE_OPENMP)
   typedef Kokkos::OpenMP       MemSpace;
#else
   typedef Kokkos::HostSpace    MemSpace;
#endif

using HostSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceSpace = Kokkos::DefaultExecutionSpace;

using DeviceShmem = DeviceSpace::scratch_memory_space;
using HostShmem = HostSpace::scratch_memory_space;
using DynamicScheduleType = Kokkos::Schedule<Kokkos::Dynamic>;
using TeamHandleType = Kokkos::TeamPolicy<HostSpace, DynamicScheduleType>::member_type;
using DeviceTeamHandleType = Kokkos::TeamPolicy<DeviceSpace, DynamicScheduleType>::member_type;

template <typename T, typename SHMEM=HostShmem>
using SharedMemView = Kokkos::View<T, Kokkos::LayoutRight, SHMEM, Kokkos::MemoryUnmanaged>;

template<typename T>
using AlignedViewType = Kokkos::View<T, Kokkos::MemoryTraits<Kokkos::Aligned>>;

using DeviceTeamPolicy = Kokkos::TeamPolicy<DeviceSpace>;
using HostTeamPolicy = Kokkos::TeamPolicy<HostSpace>;
using DeviceTeam = DeviceTeamPolicy::member_type;
using HostTeam = HostTeamPolicy::member_type;

inline HostTeamPolicy get_host_team_policy(const size_t sz, const size_t bytes_per_team,
    const size_t bytes_per_thread)
{
  HostTeamPolicy policy(sz, Kokkos::AUTO);
  return policy.set_scratch_size(1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

inline DeviceTeamPolicy get_device_team_policy(const size_t sz, const size_t bytes_per_team,
    const size_t bytes_per_thread, const int threads_per_team)
{
  DeviceTeamPolicy policy(sz, threads_per_team);
  return policy.set_scratch_size(1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

inline DeviceTeamPolicy get_device_team_policy(const size_t sz, const size_t bytes_per_team,
    const size_t bytes_per_thread)
{
  DeviceTeamPolicy policy(sz, Kokkos::AUTO);
  return policy.set_scratch_size(1, Kokkos::PerTeam(bytes_per_team), Kokkos::PerThread(bytes_per_thread));
}

template<typename T, typename TEAMHANDLETYPE, typename TeamShmemType=HostShmem>
KOKKOS_FUNCTION
SharedMemView<T*,TeamShmemType> get_shmem_view_1D(const TEAMHANDLETYPE& team, size_t len)
{
  return Kokkos::subview(SharedMemView<T**,TeamShmemType>(team.team_scratch(1), team.team_size(), len), team.team_rank(), Kokkos::ALL());
}

template<typename T, typename TEAMHANDLETYPE, typename TeamShmemType=HostShmem>
KOKKOS_FUNCTION
SharedMemView<T**,TeamShmemType> get_shmem_view_2D(const TEAMHANDLETYPE& team, size_t len1, size_t len2)
{
  return Kokkos::subview(SharedMemView<T***,TeamShmemType>(team.team_scratch(1), team.team_size(), len1, len2), team.team_rank(), Kokkos::ALL(), Kokkos::ALL());
}

template<typename T, typename TEAMHANDLETYPE, typename TeamShmemType=HostShmem>
KOKKOS_FUNCTION
SharedMemView<T***,TeamShmemType> get_shmem_view_3D(const TEAMHANDLETYPE& team, size_t len1, size_t len2, size_t len3)
{
  return Kokkos::subview(SharedMemView<T****,TeamShmemType>(team.team_scratch(1), team.team_size(), len1, len2, len3), team.team_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
}

#ifndef KOKKOS_ENABLE_CUDA
template<typename T, typename TEAMHANDLETYPE, typename TeamShmemType=HostShmem>
KOKKOS_FUNCTION
SharedMemView<T****,TeamShmemType> get_shmem_view_4D(const TEAMHANDLETYPE& team, size_t len1, size_t len2, size_t len3, size_t len4)
{
  return Kokkos::subview(SharedMemView<T*****,TeamShmemType>(team.team_scratch(1), team.team_size(), len1, len2, len3, len4), team.team_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
}

template<typename T, typename TEAMHANDLETYPE, typename TeamShmemType=HostShmem>
KOKKOS_FUNCTION
SharedMemView<T*****,TeamShmemType> get_shmem_view_5D(const TEAMHANDLETYPE& team, size_t len1, size_t len2, size_t len3, size_t len4, size_t len5)
{
  return Kokkos::subview(SharedMemView<T******,TeamShmemType>(team.team_scratch(1), team.team_size(), len1, len2, len3, len4, len5), team.team_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
}
#endif

template<typename SizeType, class Function>
void kokkos_parallel_for(const std::string& debuggingName, SizeType n, Function loop_body)
{
    Kokkos::parallel_for(debuggingName, Kokkos::RangePolicy<Kokkos::Serial>(0, n), loop_body);
}

template<typename SizeType, class Function, typename ReduceType>
void kokkos_parallel_reduce(SizeType n, Function loop_body, ReduceType& reduce, const std::string& debuggingName)
{
    Kokkos::parallel_reduce(debuggingName, Kokkos::RangePolicy<Kokkos::Serial>(0, n), loop_body, reduce);
}

template<typename T>
inline T* kokkos_malloc_on_device(const std::string& debuggingName) {
  return static_cast<T*>(Kokkos::kokkos_malloc(debuggingName, sizeof(T)));
}
inline void kokkos_free_on_device(void * ptr) { Kokkos::kokkos_free(ptr); }
}
}

#endif /* INCLUDE_KOKKOSINTERFACE_H_ */
