/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMADVDIFFELEMKERNEL_H
#define MOMENTUMADVDIFFELEMKERNEL_H

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

/** Advection diffusion term for momentum equation (velocity DOF)
 */
template<typename AlgTraits>
class MomentumAdvDiffElemKernel: public Kernel
{
public:
  MomentumAdvDiffElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    VectorFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~MomentumAdvDiffElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumAdvDiffElemKernel() = delete;

  unsigned velocityNp1_  {stk::mesh::InvalidOrdinal};
  unsigned coordinates_  {stk::mesh::InvalidOrdinal};
  unsigned viscosity_    {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};

  const int* lrscv_;

  const double includeDivU_;
  const bool shiftedGradOp_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_adv_shape_function_{"v_adv_shape_function"};
};

}  // nalu
}  // sierra

#endif /* MOMENTUMADVDIFFELEMKERNEL_H */
