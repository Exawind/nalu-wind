// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "edge_kernels/AssembleTAMSEdgeKernelAlg.h"
#include "edge_kernels/EdgeKernel.h"
#include "edge_kernels/MomentumSSTTAMSDiffEdgeKernel.h"

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
