// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SCALARFLUXBCELEMKERNEL_H
#define SCALARFLUXBCELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;

template <typename BcAlgTraits>
class ScalarFluxBCElemKernel
  : public NGPKernel<ScalarFluxBCElemKernel<BcAlgTraits>>
{
public:
  ScalarFluxBCElemKernel(
    const stk::mesh::BulkData&,
    ScalarFieldType*,
    std::string,
    const bool,
    ElemDataRequests&);

  KOKKOS_FORCEINLINE_FUNCTION
  ScalarFluxBCElemKernel() = default;

  KOKKOS_FUNCTION
  virtual ~ScalarFluxBCElemKernel() = default;

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
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned bcScalarQ_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};

  const bool useShifted_{false};

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra


#endif /* SCALARFLUXBCELEMKERNEL_H */
