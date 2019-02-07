/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALARDIFFFEMKERNEL_H
#define SCALARDIFFFEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>
#include <vector>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class Hex8FEM;
class MasterElement;
class SolutionOptions;

/** CVFEM scalar advection/diffusion kernel
 */
template<typename AlgTraits>
class ScalarDiffFemKernel: public Kernel
{
public:
  ScalarDiffFemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ScalarFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~ScalarDiffFemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  ScalarDiffFemKernel() = delete;

  const stk::mesh::BulkData* bulkData_;
  unsigned scalarQ_{stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};

  // master element
  Hex8FEM * meFEM_;
  double *ipWeight_;
  const bool shiftedGradOp_;
  
  /// Shape functions
  AlignedViewType<DoubleType[AlgTraits::numGp_][AlgTraits::nodesPerElement_]> v_shape_function_ { "v_shape_func" };
};

}  // nalu
}  // sierra

#endif /* SCALARDIFFFEMKERNEL_H */
