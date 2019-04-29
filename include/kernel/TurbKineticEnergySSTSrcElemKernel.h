/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TURBKINETICENERGYSSTSRCELEMKERNEL_H
#define TURBKINETICENERGYSSTSRCELEMKERNEL_H

#include "Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

template <typename AlgTraits>
class TurbKineticEnergySSTSrcElemKernel : public NGPKernel<TurbKineticEnergySSTSrcElemKernel<AlgTraits>>
{
public:
  TurbKineticEnergySSTSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  KOKKOS_FUNCTION TurbKineticEnergySSTSrcElemKernel() = default;

  KOKKOS_FUNCTION virtual ~TurbKineticEnergySSTSrcElemKernel() = default;

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  KOKKOS_FUNCTION
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned  tkeNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  sdrNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned  coordinates_ {stk::mesh::InvalidOrdinal};

  const bool lumpedMass_;
  const bool shiftedGradOp_;
  const double betaStar_;
  double tkeProdLimitRatio_{0.0};

  MasterElement* meSCV_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* TURBKINETICENERGYSSTSRCELEMKERNEL_H */
