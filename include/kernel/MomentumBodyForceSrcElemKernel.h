/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMBODYFORCESRCELEMKERNEL_H
#define MOMENTUMBODYFORCESRCELEMKERNEL_H

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

/** CMM body force term for momentum equation (velocity DOF)
 */
template <typename AlgTraits>
class MomentumBodyForceSrcElemKernel : public Kernel
{
public:
  MomentumBodyForceSrcElemKernel(
    const stk::mesh::BulkData&, const SolutionOptions&, ElemDataRequests&);

  virtual ~MomentumBodyForceSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumBodyForceSrcElemKernel() = delete;

  unsigned densityNp1_{stk::mesh::InvalidOrdinal};

  AlignedViewType<DoubleType[AlgTraits::nDim_]> bodyForce_{"v_bodyForce"};

  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_func"};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMBODYFORCESRCELEMKERNEL_H */
