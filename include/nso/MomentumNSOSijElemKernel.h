// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef MOMENTUMNSOSIJELEMKERNEL_H
#define MOMENTUMNSOSIJELEMKERNEL_H

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

/** NSO for momentum equation
 *
 */
template<typename AlgTraits>
class MomentumNSOSijElemKernel : public Kernel
{
public:
  MomentumNSOSijElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    VectorFieldType*,
    ElemDataRequests&);

  virtual ~MomentumNSOSijElemKernel() {}

  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumNSOSijElemKernel() = delete;

  VectorFieldType *velocityNp1_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  ScalarFieldType *pressure_{nullptr};
  VectorFieldType *velocityRTM_{nullptr};
  VectorFieldType *coordinates_{nullptr};
  VectorFieldType *Gjp_{nullptr};

  // master element
  const int *lrscv_;

  const double Cupw_{0.1};
  const double small_{1.0e-16};
  const double includeDivU_;
  const bool shiftedGradOp_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

}  // nalu
}  // sierra

#endif /* MOMENTUMNSOSIJELEMKERNEL_H */
