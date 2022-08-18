// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/AssembleAMSEdgeKernelAlg.h"
#include "edge_kernels/EdgeKernel.h"
#include "edge_kernels/MomentumSSTAMSDiffEdgeKernel.h"
#include "edge_kernels/MomentumSSTLRAMSDiffEdgeKernel.h"
#include "edge_kernels/MomentumKEAMSDiffEdgeKernel.h"
#include "edge_kernels/MomentumKOAMSDiffEdgeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

AssembleAMSEdgeKernelAlg::AssembleAMSEdgeKernelAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : AssembleEdgeKernelAlg(realm, part, eqSystem)
{

  // Register AMS Kernels directly
  if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SST_AMS)
    add_kernel<MomentumSSTAMSDiffEdgeKernel>(
      realm_.bulk_data(), *realm_.solutionOptions_);

  else if (
    realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::SSTLR_AMS)
    add_kernel<MomentumSSTLRAMSDiffEdgeKernel>(
      realm_.bulk_data(), *realm_.solutionOptions_);

  else if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::KE_AMS)
    add_kernel<MomentumKEAMSDiffEdgeKernel>(
      realm_.bulk_data(), *realm_.solutionOptions_);

  else if (realm_.solutionOptions_->turbulenceModel_ == TurbulenceModel::KO_AMS)
    add_kernel<MomentumKOAMSDiffEdgeKernel>(
      realm_.bulk_data(), *realm_.solutionOptions_);

  else
    throw std::runtime_error(
      "AssembleAMSEdgeKernelAlg: Not a valid turbulence model");
}

} // namespace nalu
} // namespace sierra
