// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifdef NALU_USES_HYPRE
#include "HYPRE_utilities.h"
#include "krylov.h"
#include "HYPRE.h"
#include "_hypre_utilities.h"

#ifdef KOKKOS_ENABLE_CUDA
#include "_hypre_utilities.hpp"
#endif

namespace nalu_hypre {
  void hypre_initialize()
  {
#ifdef KOKKOS_ENABLE_CUDA
    HYPRE_Init();
    hypre_HandleDefaultExecPolicy(hypre_handle()) = HYPRE_EXEC_DEVICE;
    hypre_HandleSpgemmUseCusparse(hypre_handle()) = 0;
#endif
  }

  void hypre_finalize()
  {
    HYPRE_Finalize();
  }
}

#endif
