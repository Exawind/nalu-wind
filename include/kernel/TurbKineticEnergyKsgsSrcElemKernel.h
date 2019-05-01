/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TurbKineticEnergyKsgsSrcElemKernel_H
#define TurbKineticEnergyKsgsSrcElemKernel_H

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

/** Add Ksgs source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class TurbKineticEnergyKsgsSrcElemKernel: public NGPKernel<TurbKineticEnergyKsgsSrcElemKernel<AlgTraits>>
{
public:
  TurbKineticEnergyKsgsSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  KOKKOS_FUNCTION TurbKineticEnergyKsgsSrcElemKernel() = default;

  KOKKOS_FUNCTION virtual ~TurbKineticEnergyKsgsSrcElemKernel() = default;

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
  unsigned tkeNp1_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolume_ {stk::mesh::InvalidOrdinal};
  unsigned Gju_ {stk::mesh::InvalidOrdinal};

  double cEps_{0.0};
  double tkeProdLimitRatio_{0.0};

  MasterElement* meSCV_{nullptr};
};

}  // nalu
}  // sierra

#endif /* TurbKineticEnergyKsgsSrcElemKernel_H */
