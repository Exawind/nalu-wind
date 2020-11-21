// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMABLWALLFUNCEDGEKERNEL_H
#define MOMENTUMABLWALLFUNCEDGEKERNEL_H

#include "kernel/Kernel.h"
#include "KokkosInterface.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
class MomentumABLWallFuncEdgeKernel: public NGPKernel<MomentumABLWallFuncEdgeKernel<BcAlgTraits>>
{
public:
  MomentumABLWallFuncEdgeKernel(
    stk::mesh::MetaData&,
    const double&,
    const double&,
    const double&,
    const double&,
    ElemDataRequests& faceDataPreReqs);

  KOKKOS_DEFAULTED_FUNCTION MomentumABLWallFuncEdgeKernel() = default;

  KOKKOS_DEFAULTED_FUNCTION virtual ~MomentumABLWallFuncEdgeKernel() = default;

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned velocityNp1_    {stk::mesh::InvalidOrdinal};
  unsigned bcVelocity_     {stk::mesh::InvalidOrdinal};
  unsigned density_        {stk::mesh::InvalidOrdinal};
  unsigned bcHeatFlux_     {stk::mesh::InvalidOrdinal};
  unsigned specificHeat_   {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_    {stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_   {stk::mesh::InvalidOrdinal};

  //! Acceleration due to gravity (m/s^2)
  const DoubleType gravity_;

  //! Roughness height (m)
  const DoubleType z0_;

  //! Reference temperature (K)
  const DoubleType Tref_;

  //! von Karman constant
  const DoubleType kappa_{0.41};
  const DoubleType beta_m_{5.0};
  const DoubleType beta_h_{5.0};
  const DoubleType gamma_m_{16.0};
  const DoubleType gamma_h_{16.0};

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra


#endif /* MOMENTUMABLWALLFUNCEDGEKERNEL_H */
