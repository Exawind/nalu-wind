/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WALLDISTELEMKERNEL_H
#define WALLDISTELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"
#include "Kokkos_Core.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

template<typename AlgTraits>
class WallDistElemKernel : public NGPKernel<WallDistElemKernel<AlgTraits>>
{
public:
  WallDistElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  KOKKOS_FUNCTION WallDistElemKernel() = default;

  KOKKOS_FUNCTION virtual ~WallDistElemKernel() = default;

  using Kernel::execute;
  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};

  MasterElement* meSCS_{nullptr};
  MasterElement* meSCV_{nullptr};

  const bool shiftPoisson_{false};
};

}  // nalu
}  // sierra


#endif /* WALLDISTELEMKERNEL_H */
