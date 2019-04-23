/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALARMASSELEMKERNEL_H
#define SCALARMASSELEMKERNEL_H

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

/** CMM (BDF2/BE) for scalar equation
 */
template<typename AlgTraits>
class ScalarMassElemKernel: public NGPKernel<ScalarMassElemKernel<AlgTraits>>
{
public:
  ScalarMassElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ElemDataRequests&,
    const bool lumpedMass);

  KOKKOS_FUNCTION ScalarMassElemKernel() = default;

  virtual ~ScalarMassElemKernel() = default;

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
  unsigned scalarQNm1_ {stk::mesh::InvalidOrdinal};
  unsigned scalarQN_ {stk::mesh::InvalidOrdinal};
  unsigned scalarQNp1_ {stk::mesh::InvalidOrdinal};
  unsigned densityNm1_ {stk::mesh::InvalidOrdinal};
  unsigned densityN_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};

  double dt_{0.0};
  double gamma1_{0.0};
  double gamma2_{0.0};
  double gamma3_{0.0};
  double diagRelaxFactor_{1.0};
  const bool lumpedMass_;

  MasterElement* meSCV_ {nullptr};
};

}  // nalu
}  // sierra

#endif /* SCALARMASSELEMKERNEL_H */
