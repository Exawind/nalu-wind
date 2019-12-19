// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMOPENEDGEKERNEL_h
#define MOMENTUMOPENEDGEKERNEL_h

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra{
namespace nalu{

class SolutionOptions;
class TimeIntegrator;

template<typename BcAlgTraits>
class MomentumOpenEdgeKernel : public NGPKernel<MomentumOpenEdgeKernel<BcAlgTraits>>
{
public:

  MomentumOpenEdgeKernel(
    const stk::mesh::MetaData&,
    SolutionOptions*,
    ScalarFieldType*,
    ElemDataRequests&,
    ElemDataRequests&);

  KOKKOS_FORCEINLINE_FUNCTION
  MomentumOpenEdgeKernel() = default;

  KOKKOS_INLINE_FUNCTION
  virtual ~MomentumOpenEdgeKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    int);

private:

  const unsigned coordinates_      {stk::mesh::InvalidOrdinal};
  const unsigned dudx_             {stk::mesh::InvalidOrdinal};
  const unsigned exposedAreaVec_   {stk::mesh::InvalidOrdinal};
  const unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
  const unsigned velocityBc_       {stk::mesh::InvalidOrdinal};
  const unsigned velocityNp1_      {stk::mesh::InvalidOrdinal};
  const unsigned viscosity_        {stk::mesh::InvalidOrdinal};

  const double includeDivU_;
  const double nfEntrain_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace Sierra

#endif /* MOMENTUMOPENEDGEKERNEL_h */
