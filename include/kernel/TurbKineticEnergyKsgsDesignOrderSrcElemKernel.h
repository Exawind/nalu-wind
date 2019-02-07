/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TurbKineticEnergyKsgsDesignOrderSrcElemKernel_H
#define TurbKineticEnergyKsgsDesignOrderSrcElemKernel_H

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

/** Add Ksgs source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class TurbKineticEnergyKsgsDesignOrderSrcElemKernel: public Kernel
{
public:
  TurbKineticEnergyKsgsDesignOrderSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  virtual ~TurbKineticEnergyKsgsDesignOrderSrcElemKernel();
  
  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TurbKineticEnergyKsgsDesignOrderSrcElemKernel() = delete;
  
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned tkeNp1_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolume_ {stk::mesh::InvalidOrdinal};

  const double cEps_;
  const double tkeProdLimitRatio_;
  
  /// Integration point to node mapping
  const int* ipNodeMap_;
  
  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};
 
}  // nalu
}  // sierra

#endif /* TurbKineticEnergyKsgsDesignOrderSrcElemKernel_H */
