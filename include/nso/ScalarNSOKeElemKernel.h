/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALARNSOKEELEMKERNEL_H
#define SCALARNSOKEELEMKERNEL_H

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

/** NSO ke for scalar
 *
 */
template<typename AlgTraits>
class ScalarNSOKeElemKernel : public Kernel
{
public:
  ScalarNSOKeElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions& solnOpts,
    ScalarFieldType* scalarQ,
    VectorFieldType* Gjq,
    const double turbCoeff,
    const double fourthFac,
    ElemDataRequests& dataPreReqs);

  virtual ~ScalarNSOKeElemKernel() {}

  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  ScalarNSOKeElemKernel() = delete;

  ScalarFieldType *scalarQ_{nullptr};
  VectorFieldType *Gjq_{nullptr};
  VectorFieldType *coordinates_{nullptr};
  VectorFieldType *velocityNp1_{nullptr};
  VectorFieldType *velocityRTM_{nullptr};
  VectorFieldType *Gjp_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  ScalarFieldType *pressure_{nullptr};

  // master element
  const int *lrscv_;
  const double fourthFac_;
  const double turbCoeff_;
  const double Cupw_{0.1};
  const double small_{1.0e-16};
  const bool shiftedGradOp_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

}  // nalu
}  // sierra

#endif /* SCALARNSOKEELEMKERNEL_H */
