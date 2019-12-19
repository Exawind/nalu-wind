// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef STEADYTHERMAL3DCONTACTSRCELEMKERNEL_H
#define STEADYTHERMAL3DCONTACTSRCELEMKERNEL_H

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

template<typename AlgTraits>
class SteadyThermal3dContactSrcElemKernel: public Kernel
{
public:
  SteadyThermal3dContactSrcElemKernel(
    const stk::mesh::BulkData&,
    SolutionOptions&,
    ElemDataRequests&);

  virtual ~SteadyThermal3dContactSrcElemKernel() {}

  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  SteadyThermal3dContactSrcElemKernel() = delete;

  VectorFieldType *coordinates_;

  const int *ipNodeMap_;

  const double a_;
  const double k_;
  const double pi_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

}  // nalu
}  // sierra

#endif /* STEADYTHERMAL3DCONTACTSRCELEMKERNEL_H */
