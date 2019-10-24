/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corp.                                           */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ContinuityInflowElemKernel_h
#define ContinuityInflowElemKernel_h

#include "FieldTypeDef.h"
#include "kernel/Kernel.h"

#include <stk_mesh/base/BulkData.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;
class TimeIntegrator;

/** Add Int rho*uj*nj*dS
 */
template<typename BcAlgTraits>
class ContinuityInflowElemKernel: public NGPKernel<ContinuityInflowElemKernel<BcAlgTraits>>
{
public:
  ContinuityInflowElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions &solnOpts,
    const bool &useShifted,
    ElemDataRequests &faceDataPreReqs);

  KOKKOS_FUNCTION
  virtual ~ContinuityInflowElemKernel() = default;

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType **, DeviceShmem>&lhs,
    SharedMemView<DoubleType *, DeviceShmem>&rhs,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews);

private:
  ContinuityInflowElemKernel() = delete;

  unsigned velocityBC_ {stk::mesh::InvalidOrdinal};
  unsigned densityBC_ {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};

  const bool useShifted_;
  double projTimeScale_;
  const double interpTogether_;
  const double om_interpTogether_;

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra

#endif /* ContinuityInflowElemKernel_h */
