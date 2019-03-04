/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WALLDISTELEMKERNEL_H
#define WALLDISTELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"
#include "Kokkos_Core.hpp"

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

template<typename AlgTraits>
class WallDistElemKernel : public Kernel
{
public:
  WallDistElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  virtual ~WallDistElemKernel();

  virtual void setup(const TimeIntegrator&);

  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  WallDistElemKernel() = delete;
  WallDistElemKernel(const WallDistElemKernel&) = delete;

  unsigned coordinates_ {stk::mesh::InvalidOrdinal};

  // work arrays
  Kokkos::View<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"view_shape_function"};

  const int* lrscv_;
  const int* ipNodeMap_;

  const bool shiftPoisson_{false};
};

}  // nalu
}  // sierra


#endif /* WALLDISTELEMKERNEL_H */
