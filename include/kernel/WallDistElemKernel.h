// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
