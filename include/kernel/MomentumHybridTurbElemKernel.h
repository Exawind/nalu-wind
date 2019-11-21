// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMHYBRIDTURBELEMKERNEL_H
#define MOMENTUMHYBRIDTURBELEMKERNEL_H

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

/** Hybrid turbulence for momentum equation
 *
 */
template <typename AlgTraits>
class MomentumHybridTurbElemKernel : public NGPKernel<MomentumHybridTurbElemKernel<AlgTraits>>
{
public:
  MomentumHybridTurbElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    VectorFieldType*,
    ElemDataRequests&);

  KOKKOS_FUNCTION MomentumHybridTurbElemKernel() = default;

  KOKKOS_FUNCTION virtual ~MomentumHybridTurbElemKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned  velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  tkeNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  alphaNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  mutij_ {stk::mesh::InvalidOrdinal};
  unsigned  coordinates_ {stk::mesh::InvalidOrdinal};

  const bool shiftedGradOp_;

  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMHYBRIDTURBELEMKERNEL_H */
