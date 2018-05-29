/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMHYBRIDTURBELEMKERNEL_H
#define MOMENTUMHYBRIDTURBELEMKERNEL_H

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

/** Hybrid turbulence for momentum equation
 *
 */
template <typename AlgTraits>
class MomentumHybridTurbElemKernel : public Kernel
{
public:
  MomentumHybridTurbElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    VectorFieldType*,
    ElemDataRequests&);

  virtual ~MomentumHybridTurbElemKernel() {}

  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumHybridTurbElemKernel() = delete;

  VectorFieldType* velocityNp1_{nullptr};
  ScalarFieldType* densityNp1_{nullptr};
  ScalarFieldType* tkeNp1_{nullptr};
  ScalarFieldType* alphaNp1_{nullptr};
  GenericFieldType* mutij_{nullptr};
  VectorFieldType* coordinates_{nullptr};

  // master element
  const int* lrscv_;

  const bool shiftedGradOp_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMHYBRIDTURBELEMKERNEL_H */
