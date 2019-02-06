/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "edge_kernels/AssembleTAMSEdgeKernelAlg.h"
#include "edge_kernels/EdgeKernel.h"

namespace sierra {
namespace nalu {

AssembleTAMSEdgeKernelAlg::AssembleTAMSEdgeKernelAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : AssembleEdgeKernelAlg(realm, part, eqSystem)
{
  // Register TAMS Kernels directly
  add_kernel<MomentumSSTTAMSDiffEdgeKernel>(
    realm_.bulk_data(), *realm_.solutionOptions_);
}

} // namespace nalu
} // namespace sierra
