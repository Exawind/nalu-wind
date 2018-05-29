/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef RadTransIsotropicScatteringElemKernel_H
#define RadTransIsotropicScatteringElemKernel_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;

/** Add ((abs+scat)*I - radSrc)*dVol source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class RadTransIsotropicScatteringElemKernel: public Kernel
{
public:
  RadTransIsotropicScatteringElemKernel(
      const stk::mesh::BulkData&,
      const bool lumpedMass,
      ElemDataRequests&);

  virtual ~RadTransIsotropicScatteringElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  RadTransIsotropicScatteringElemKernel() = delete;

  ScalarFieldType *scalarFlux_{nullptr};
  ScalarFieldType *scattering_{nullptr};
  
  // 1/(4.0*pi)
  const double invFourPi_;

  /// Integration point to node mapping
  const int* ipNodeMap_;

  // fixed scratch space
  AlignedViewType<DoubleType[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
};

}  // nalu
}  // sierra

#endif /* RadTransIsotropicScatteringElemKernel_H */
