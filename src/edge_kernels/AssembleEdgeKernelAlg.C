// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/AssembleEdgeKernelAlg.h"
#include "edge_kernels/EdgeKernel.h"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

AssembleEdgeKernelAlg::AssembleEdgeKernelAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
}

AssembleEdgeKernelAlg::~AssembleEdgeKernelAlg()
{
  // Release device pointers if any
  for (auto& kern : edgeKernels_)
    kern->free_on_device();
}

void
AssembleEdgeKernelAlg::execute()
{
  const size_t numKernels = edgeKernels_.size();
  if (numKernels < 1)
    return;

  for (auto& kern : edgeKernels_)
    kern->setup(realm_);

  auto ngpKernels = nalu_ngp::create_ngp_view<EdgeKernel>(edgeKernels_);

  run_algorithm(
    realm_.bulk_data(), KOKKOS_LAMBDA(
                          EdgeKernelTraits::ShmemDataType & smdata,
                          const stk::mesh::FastMeshIndex& edge,
                          const stk::mesh::FastMeshIndex& nodeL,
                          const stk::mesh::FastMeshIndex& nodeR) {
      for (size_t i = 0; i < numKernels; i++) {
        EdgeKernel* kernel = ngpKernels(i);
        kernel->execute(smdata, edge, nodeL, nodeR);
      }
    });
}

} // namespace nalu
} // namespace sierra
