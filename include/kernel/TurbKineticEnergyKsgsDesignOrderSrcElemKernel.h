// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef TurbKineticEnergyKsgsDesignOrderSrcElemKernel_H
#define TurbKineticEnergyKsgsDesignOrderSrcElemKernel_H

#include "Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** Add Ksgs source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class TurbKineticEnergyKsgsDesignOrderSrcElemKernel: public NGPKernel<TurbKineticEnergyKsgsDesignOrderSrcElemKernel<AlgTraits>>
{
public:
  TurbKineticEnergyKsgsDesignOrderSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  KOKKOS_FUNCTION TurbKineticEnergyKsgsDesignOrderSrcElemKernel() = default;

  KOKKOS_FUNCTION virtual ~TurbKineticEnergyKsgsDesignOrderSrcElemKernel() = default;
  
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
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolume_ {stk::mesh::InvalidOrdinal};

  const double cEps_;
  const double tkeProdLimitRatio_;

  MasterElement* meSCV_{nullptr};
};
 
}  // nalu
}  // sierra

#endif /* TurbKineticEnergyKsgsDesignOrderSrcElemKernel_H */
