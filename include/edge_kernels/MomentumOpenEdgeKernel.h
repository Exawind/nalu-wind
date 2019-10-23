/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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

  // TODO: This needs to go away
  SolutionOptions* solnOpts_{nullptr};

  const unsigned coordinates_      {stk::mesh::InvalidOrdinal};
  const unsigned dudx_             {stk::mesh::InvalidOrdinal};
  const unsigned exposedAreaVec_   {stk::mesh::InvalidOrdinal};
  const unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
  const unsigned velocityBc_       {stk::mesh::InvalidOrdinal};
  const unsigned velocityNp1_      {stk::mesh::InvalidOrdinal};
  const unsigned viscosity_        {stk::mesh::InvalidOrdinal};

  const double includeDivU_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace Sierra

#endif /* MOMENTUMOPENEDGEKERNEL_h */
