/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MOMENTUMSYMMETRYELEMKERNEL_H
#define MOMENTUMSYMMETRYELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class ElemDataRequests;
class MasterElement;

/** Symmetry kernel for momentum equation (velocity DOF)
 */
template<typename BcAlgTraits>
class MomentumSymmetryElemKernel: public Kernel
{
public:
  MomentumSymmetryElemKernel(
    const stk::mesh::MetaData &metaData,
    const SolutionOptions &solnOpts,
    VectorFieldType *velocity,
    ScalarFieldType *viscosity,
    ElemDataRequests &faceDataPreReqs,
    ElemDataRequests &elemDataPreReqs);

  virtual ~MomentumSymmetryElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**> &lhs,
    SharedMemView<DoubleType*> &rhs,
    ScratchViews<DoubleType> &faceScratchViews,
    ScratchViews<DoubleType> &elemScratchViews,
    int elemFaceOrdinal);

private:
  MomentumSymmetryElemKernel() = delete;

  unsigned viscosity_      {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_    {stk::mesh::InvalidOrdinal};
  unsigned coordinates_    {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};

  const double includeDivU_;
  const bool shiftedGradOp_;

  MasterElement *meSCS_{nullptr};

  /// Shape functions
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_shape_function_ {"view_face_shape_func"};
};

}  // nalu
}  // sierra

#endif /* MOMENTUMSYMMETRYELEMKERNEL_H */
