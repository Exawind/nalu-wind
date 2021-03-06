// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef CONTINUITYADVELEMKERNEL_H
#define CONTINUITYADVELEMKERNEL_H

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

/** CMM (BDF2) for continuity equation (pressure DOF)
 */
template<typename AlgTraits>
class ContinuityAdvElemKernel: public NGPKernel<ContinuityAdvElemKernel<AlgTraits>>
{
public:
  ContinuityAdvElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  KOKKOS_DEFAULTED_FUNCTION
  ContinuityAdvElemKernel() = default;

  virtual ~ContinuityAdvElemKernel() = default;

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&);

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
  // extract fields; nodal
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned Gpdx_ {stk::mesh::InvalidOrdinal};
  unsigned pressure_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned Udiag_ {stk::mesh::InvalidOrdinal};

  double projTimeScale_{1.0};

  const bool meshMotion_;
  const bool shiftMdot_;
  const bool shiftPoisson_;
  const bool reducedSensitivities_;
  const double interpTogether_;
  const double om_interpTogether_;

  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra

#endif /* CONTINUITYADVELEMKERNEL_H */
