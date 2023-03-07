// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef CONTINUITYOPENEDGEKERNEL_H
#define CONTINUITYOPENEDGEKERNEL_H

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;
class TimeIntegrator;

template <typename BcAlgTraits>
class ContinuityOpenEdgeKernel
  : public NGPKernel<ContinuityOpenEdgeKernel<BcAlgTraits>>
{
public:
  ContinuityOpenEdgeKernel(
    const stk::mesh::MetaData&,
    SolutionOptions*,
    ElemDataRequests&,
    ElemDataRequests&);

  KOKKOS_DEFAULTED_FUNCTION
  ContinuityOpenEdgeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~ContinuityOpenEdgeKernel() = default;

  virtual void setup(const TimeIntegrator&);

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    int);

private:
  // TODO: This needs to go away
  SolutionOptions* solnOpts_{nullptr};

  const unsigned coordinates_{stk::mesh::InvalidOrdinal};
  const unsigned velocityRTM_{stk::mesh::InvalidOrdinal};
  const unsigned pressure_{stk::mesh::InvalidOrdinal};
  const unsigned density_{stk::mesh::InvalidOrdinal};
  const unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  const unsigned pressureBC_{stk::mesh::InvalidOrdinal};
  const unsigned Gpdx_{stk::mesh::InvalidOrdinal};
  const unsigned Udiag_{stk::mesh::InvalidOrdinal};
  const unsigned dynPress_{stk::mesh::InvalidOrdinal};

  DoubleType tauScale_;
  DoubleType mdotCorr_;
  const DoubleType pstabFac_;
  const DoubleType nocFac_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};

  const DoubleType solveInc_;
};

} // namespace nalu
} // namespace sierra

#endif /* CONTINUITYOPENEDGEKERNEL_H */
