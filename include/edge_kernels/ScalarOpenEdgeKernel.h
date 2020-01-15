// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SCALAROPENEDGEKERNEL_H
#define SCALAROPENEDGEKERNEL_H

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class SolutionOptions;

template<typename BcAlgTraits>
class ScalarOpenEdgeKernel : public NGPKernel<ScalarOpenEdgeKernel<BcAlgTraits>>
{
public:
  ScalarOpenEdgeKernel(
    const stk::mesh::MetaData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  KOKKOS_FORCEINLINE_FUNCTION
  ScalarOpenEdgeKernel() = default;

  KOKKOS_INLINE_FUNCTION
  virtual ~ScalarOpenEdgeKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned scalarQ_ { stk::mesh::InvalidOrdinal };
  unsigned bcScalarQ_ { stk::mesh::InvalidOrdinal };
  unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};

  const DoubleType relaxFac_;

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra


#endif /* SCALAROPENEDGEKERNEL_H */
