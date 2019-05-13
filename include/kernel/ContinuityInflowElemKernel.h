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
class ContinuityInflowElemKernel: public Kernel
{
public:
  ContinuityInflowElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions &solnOpts,
    const bool &useShifted,
    ElemDataRequests &faceDataPreReqs);

  virtual ~ContinuityInflowElemKernel();

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType **>&lhs,
    SharedMemView<DoubleType *>&rhs,
    ScratchViews<DoubleType>& scratchViews);

private:
  ContinuityInflowElemKernel() = delete;

  unsigned velocityBC_ {stk::mesh::InvalidOrdinal};
  unsigned densityBC_ {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};

  const bool useShifted_;
  double projTimeScale_;
  const double interpTogether_;
  const double om_interpTogether_;

  // Integration point to node mapping 
  const int *ipNodeMap_{nullptr};

  // scratch space
  Kokkos::View<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> 
    vf_shape_function_{"vf_shape_function"};
};

}  // nalu
}  // sierra

#endif /* ContinuityInflowElemKernel_h */
