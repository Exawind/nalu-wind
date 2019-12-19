// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
  using Kernel::execute;
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
