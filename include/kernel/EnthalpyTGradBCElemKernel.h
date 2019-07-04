/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ENTHALPYTGRADBCELEMKERNEL_H
#define ENTHALPYTGRADBCELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;

/** Enforce a prescribed normal temperature gradient at the boundary
 *
 *  This class handles both element and edge based BC implementation. To use
 *  this with edge-based schemes, pass `useShifted = true`.
 */
template <typename BcAlgTraits>
class EnthalpyTGradBCElemKernel
  : public NGPKernel<EnthalpyTGradBCElemKernel<BcAlgTraits>>
{
public:
  EnthalpyTGradBCElemKernel(
    const stk::mesh::BulkData&,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    std::string,
    bool,
    ElemDataRequests&);

  KOKKOS_FORCEINLINE_FUNCTION
  EnthalpyTGradBCElemKernel() = default;

  KOKKOS_FUNCTION
  virtual ~EnthalpyTGradBCElemKernel() = default;

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;

  KOKKOS_FUNCTION
  virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned bcTGrad_ {stk::mesh::InvalidOrdinal};
  unsigned evisc_ {stk::mesh::InvalidOrdinal};
  unsigned specificHeat_ {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};

  const bool useShifted_;

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra



#endif /* ENTHALPYTGRADBCELEMKERNEL_H */
