// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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

  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  MomentumCoriolisSrcElemKernel() = delete;

  CoriolisSrc cor_;
  unsigned velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};

  const int* ipNodeMap_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

} // namespace nalu
} // namespace Sierra

#endif
