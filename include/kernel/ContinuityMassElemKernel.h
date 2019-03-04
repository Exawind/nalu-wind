/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CONTINUITYMASSELEMKERNEL_H
#define CONTINUITYMASSELEMKERNEL_H

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

/** CMM (BDF2/BE) for continuity equation (pressure DOF)
 */
template<typename AlgTraits>
class ContinuityMassElemKernel: public Kernel
{
public:
  ContinuityMassElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    const bool);

  virtual ~ContinuityMassElemKernel();

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  ContinuityMassElemKernel() = delete;

  unsigned densityNm1_ {stk::mesh::InvalidOrdinal};
  unsigned densityN_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};

  double dt_{0.0};
  double gamma1_{0.0};
  double gamma2_{0.0};
  double gamma3_{0.0};
  const bool lumpedMass_;

  /// Integration point to node mapping
  const int* ipNodeMap_;

  /// Shape functions
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_ {"view_shape_func"};
};

}  // nalu
}  // sierra

#endif /* CONTINUITYMASSELEMKERNEL_H */
