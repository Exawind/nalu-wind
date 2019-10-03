/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumCoriolisSrcElemKernel_h
#define MomentumCoriolisSrcElemKernel_h

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"
#include "AlgTraits.h"
#include "CoriolisSrc.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>


namespace sierra{
namespace nalu{

template<typename AlgTraits>
class MomentumCoriolisSrcElemKernel: public NGPKernel<MomentumCoriolisSrcElemKernel<AlgTraits>>
{
public:
  MomentumCoriolisSrcElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    VectorFieldType* velocity,
    ElemDataRequests& dataPreReqs,
    bool lumped);

  virtual ~MomentumCoriolisSrcElemKernel() = default;

  using Kernel::execute;
  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  KOKKOS_FUNCTION MomentumCoriolisSrcElemKernel() = delete;

  CoriolisSrc cor_;
  unsigned velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};

  MasterElement *meSCV_{nullptr};

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace Sierra

#endif /* MomentumCoriolisSrcElemKernel_h */
