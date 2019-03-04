/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMACTUATORSRCELEMKERNEL_H
#define MOMENTUMACTUATORSRCELEMKERNEL_H

#include "Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** CMM buoyancy term for momentum equation (velocity DOF)
 */
template<typename AlgTraits>
class MomentumActuatorSrcElemKernel: public Kernel
{
public:
  MomentumActuatorSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&,
    bool lumped);

  virtual ~MomentumActuatorSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumActuatorSrcElemKernel() = delete;

  unsigned actuator_source_     {stk::mesh::InvalidOrdinal};
  unsigned actuator_source_lhs_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_         {stk::mesh::InvalidOrdinal};

  const int* ipNodeMap_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_ { "v_shape_func" };
};

}  // nalu
}  // sierra

#endif /* MOMENTUMACTUATORSRCELEMKERNEL_H */
