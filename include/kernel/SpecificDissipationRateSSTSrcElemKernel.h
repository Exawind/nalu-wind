/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SPECIFICDISSIPATIONRATESSTSRCELEMKERNEL_H
#define SPECIFICDISSIPATIONRATESSTSRCELEMKERNEL_H

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
class SpecificDissipationRateSSTSrcElemKernel : public NGPKernel<SpecificDissipationRateSSTSrcElemKernel<AlgTraits>>
{
public:
  SpecificDissipationRateSSTSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  KOKKOS_FUNCTION SpecificDissipationRateSSTSrcElemKernel() = default;

  KOKKOS_FUNCTION virtual ~SpecificDissipationRateSSTSrcElemKernel() = default;

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
  unsigned  tkeNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  sdrNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned  tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned  fOneBlend_ {stk::mesh::InvalidOrdinal};
  unsigned  coordinates_ {stk::mesh::InvalidOrdinal};

  const bool lumpedMass_;
  const bool shiftedGradOp_;
  const double betaStar_;
  const double sigmaWTwo_;
  const double betaOne_;
  const double betaTwo_;
  const double gammaOne_;
  const double gammaTwo_;
  double tkeProdLimitRatio_{0.0};

  MasterElement* meSCV_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* SPECIFICDISSIPATIONRATESSTSRCELEMKERNEL_H */
