// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMABLWALLSHEARSTRESSEDGEKERNEL_H
#define MOMENTUMABLWALLSHEARSTRESSEDGEKERNEL_H

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {


/**
 * This class applies the computed wall shear stress field to the boundary.
 *
 */
template<typename BcAlgTraits>
class MomentumABLWallShearStressEdgeKernel: public NGPKernel<MomentumABLWallShearStressEdgeKernel<BcAlgTraits>>
{
public:
  MomentumABLWallShearStressEdgeKernel(
    bool slip,
    stk::mesh::MetaData&,
    ElemDataRequests&,
    ElemDataRequests&);

  KOKKOS_DEFAULTED_FUNCTION MomentumABLWallShearStressEdgeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION virtual ~MomentumABLWallShearStressEdgeKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&,
    int);

private:
  bool slip_{true};
  unsigned exposedAreaVec_  {stk::mesh::InvalidOrdinal};
  unsigned wallShearStress_ {stk::mesh::InvalidOrdinal};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* MOMENTUMABLWALLFUNCEDGEKERNEL_H */
