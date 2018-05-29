/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumCoriolisSrcElemKernel_h
#define MomentumCoriolisSrcElemKernel_h

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"
#include "AlgTraits.h"
#include "CoriolisSrc.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>


namespace sierra{
namespace nalu{

template<typename AlgTraits>
class MomentumCoriolisSrcElemKernel: public Kernel
{
public:
  MomentumCoriolisSrcElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    VectorFieldType* velocity,
    ElemDataRequests& dataPreReqs,
    bool lumped);

  virtual ~MomentumCoriolisSrcElemKernel();

  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumCoriolisSrcElemKernel() = delete;

  CoriolisSrc cor_;
  VectorFieldType *velocityNp1_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  VectorFieldType *coordinates_{nullptr};

  const int* ipNodeMap_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace Sierra

#endif
