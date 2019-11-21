// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMADVDIFFELEMKERNEL_H
#define MOMENTUMADVDIFFELEMKERNEL_H

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

/** Advection diffusion term for momentum equation (velocity DOF)
 */
template<typename AlgTraits>
class MomentumAdvDiffElemKernel: public NGPKernel<MomentumAdvDiffElemKernel<AlgTraits>>
{
public:
  MomentumAdvDiffElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    VectorFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  KOKKOS_FUNCTION MomentumAdvDiffElemKernel() = default;

  virtual ~MomentumAdvDiffElemKernel() = default;

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned velocityNp1_  {stk::mesh::InvalidOrdinal};
  unsigned coordinates_  {stk::mesh::InvalidOrdinal};
  unsigned viscosity_    {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};

  const double includeDivU_;
  const bool shiftedGradOp_;
  const bool skewSymmetric_;

  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra

#endif /* MOMENTUMADVDIFFELEMKERNEL_H */
