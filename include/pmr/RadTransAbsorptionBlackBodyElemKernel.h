/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef RadTransAbsorptionBlackBodyElemKernel_H
#define RadTransAbsorptionBlackBodyElemKernel_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;

/** Add ((mu_a+mu_s)*I - radSrc)*dVol source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class RadTransAbsorptionBlackBodyElemKernel: public Kernel
{
public:
  RadTransAbsorptionBlackBodyElemKernel(
      const stk::mesh::BulkData&,
      const bool lumpedMass,
      ElemDataRequests&);

  virtual ~RadTransAbsorptionBlackBodyElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  RadTransAbsorptionBlackBodyElemKernel() = delete;

  ScalarFieldType *intensity_{nullptr};
  ScalarFieldType *absorption_{nullptr};
  ScalarFieldType *scattering_{nullptr};
  ScalarFieldType *radiationSource_{nullptr};

  /// Integration point to node mapping
  const int* ipNodeMap_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

}  // nalu
}  // sierra

#endif /* RadTransAbsorptionBlackBodyElemKernel_H */
