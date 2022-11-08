// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef HYPRENGP_H
#define HYPRENGP_H

#ifdef NALU_USES_HYPRE
#include "HYPRE_config.h"
#endif

#ifdef HYPRE_USING_GPU
#include "HYPRE_utilities.h"
#include "krylov.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#endif

#include "NaluParsing.h"
#include <yaml-cpp/yaml.h>

namespace nalu_hypre {

#ifdef HYPRE_USING_GPU

inline void
hypre_initialize()
{
  HYPRE_Init();
}

inline void
hypre_set_params(YAML::Node nodes)
{
#ifdef HYPRE_USING_DEVICE_POOL
  /* device pool allocator */
  hypre_uint mempool_bin_growth = 8, mempool_min_bin = 3, mempool_max_bin = 9;
  size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;
#endif
  bool use_vendor_spgemm = false;
  bool use_vendor_spmv = false;
  bool use_vendor_sptrans = false;

  const YAML::Node node = nodes["hypre_config"];
  if (node) {
#ifdef HYPRE_USING_DEVICE_POOL
    int memory_pool_mbs = 2000;
    sierra::nalu::get_if_present(
      node, "memory_pool_mbs", memory_pool_mbs, memory_pool_mbs);
    mempool_max_cached_bytes = ((size_t)memory_pool_mbs) * 1024 * 1024;
#endif

    sierra::nalu::get_if_present(
      node, "use_vendor_spgemm", use_vendor_spgemm, use_vendor_spgemm);
    sierra::nalu::get_if_present(
      node, "use_vendor_spmv", use_vendor_spmv, use_vendor_spmv);
    sierra::nalu::get_if_present(
      node, "use_vendor_sptrans", use_vendor_sptrans, use_vendor_sptrans);
  }

#ifdef HYPRE_USING_DEVICE_POOL
  /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Init
   */
  HYPRE_SetGPUMemoryPoolSize(
    mempool_bin_growth, mempool_min_bin, mempool_max_bin,
    mempool_max_cached_bytes);
#endif

  HYPRE_SetSpGemmUseVendor(use_vendor_spgemm);
  HYPRE_SetSpMVUseVendor(use_vendor_spmv);
  HYPRE_SetSpTransUseVendor(use_vendor_sptrans);
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  HYPRE_SetUseGpuRand(true);
}

inline void
hypre_set_params()
{
#ifdef HYPRE_USING_DEVICE_POOL
  /* device pool allocator */
  hypre_uint mempool_bin_growth = 8, mempool_min_bin = 3, mempool_max_bin = 9;
  size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;

  /* To be effective, hypre_SetCubMemPoolSize must immediately follow HYPRE_Init
   */
  HYPRE_SetGPUMemoryPoolSize(
    mempool_bin_growth, mempool_min_bin, mempool_max_bin,
    mempool_max_cached_bytes);
#endif

  HYPRE_SetSpGemmUseVendor(false);
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  HYPRE_SetUseGpuRand(true);
}

inline void
hypre_finalize()
{
  HYPRE_Finalize();
}

#else

inline void
hypre_initialize()
{
}

inline void
hypre_set_params(YAML::Node nodes)
{
}

inline void
hypre_set_params()
{
}

inline void
hypre_finalize()
{
}

#endif
} // namespace nalu_hypre

#endif /* HYPRENGP_H */
