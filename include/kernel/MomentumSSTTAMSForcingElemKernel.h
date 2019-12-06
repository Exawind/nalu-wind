// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMSSTTAMSFORCINGELEMKERNEL_H
#define MOMENTUMSSTTAMSFORCINGELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** Forcing stress for TAMS in momentum equation
 *
 */
template <typename AlgTraits>
class MomentumSSTTAMSForcingElemKernel : public NGPKernel<MomentumSSTTAMSForcingElemKernel<AlgTraits>>
{
public:
  MomentumSSTTAMSForcingElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  KOKKOS_FUNCTION MomentumSSTTAMSForcingElemKernel() = default;

  KOKKOS_FUNCTION virtual ~MomentumSSTTAMSForcingElemKernel() = default;

  // Perform pre-timestep work for the computational kernel
  virtual void setup(const TimeIntegrator&);

  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  double time_{0.0};
  double dt_{0.0};

  double pi_;

  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned densityNp1_{stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_{stk::mesh::InvalidOrdinal};
  unsigned sdrNp1_{stk::mesh::InvalidOrdinal};
  unsigned alpha_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned Mij_{stk::mesh::InvalidOrdinal};
  unsigned avgResAdeq_{stk::mesh::InvalidOrdinal};
  unsigned minDist_{stk::mesh::InvalidOrdinal};
  unsigned avgVelocity_{stk::mesh::InvalidOrdinal};
  unsigned avgTime_{stk::mesh::InvalidOrdinal};

  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned turbViscosity_{stk::mesh::InvalidOrdinal};

  // master element
  const double betaStar_;
  const double cMu_;
  const double forceCl_;
  const double Ceta_;
  const double Ct_;
  const double blT_;
  const double blKol_;
  const double forceFactor_;

  MasterElement* meSCV_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMSSTTAMSFORCINGELEMKERNEL_H */
