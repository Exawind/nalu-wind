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

#ifdef HYPRE_USING_CUDA
#include "HYPRE_utilities.h"
#include "krylov.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#endif

namespace nalu_hypre {

#ifdef HYPRE_USING_CUDA

inline void hypre_initialize()
{
  HYPRE_Init();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  HYPRE_SetUseGpuRand(true);
}

inline void hypre_finalize()
{
  HYPRE_Finalize();
}

#else

inline void hypre_initialize() {
}
inline void hypre_finalize() {
}

#endif
}

#endif /* HYPRENGP_H */
