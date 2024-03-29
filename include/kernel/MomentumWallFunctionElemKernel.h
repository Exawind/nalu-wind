// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MomentumWallFunctionElemKernel_h
#define MomentumWallFunctionElemKernel_h

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** Wall function approach momentum equation (velocity DOF)
 */
template <typename BcAlgTraits>
class MomentumWallFunctionElemKernel : public Kernel
{
public:
  MomentumWallFunctionElemKernel(
    const stk::mesh::BulkData&, const SolutionOptions&, ElemDataRequests&);

  virtual ~MomentumWallFunctionElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumWallFunctionElemKernel() = delete;

  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned wallFrictionVelocityBip_{stk::mesh::InvalidOrdinal};
  unsigned wallNormalDistanceBip_{stk::mesh::InvalidOrdinal};

  // turbulence model constants (constant over time and bc surfaces)
  const double elog_;
  const double kappa_;
  const double yplusCrit_;

  // Integration point to node mapping
  const int* ipNodeMap_{nullptr};

  // fixed scratch space
  AlignedViewType<
    DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]>
    vf_shape_function_{"vf_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* MomentumWallFunctionElemKernel_h */
