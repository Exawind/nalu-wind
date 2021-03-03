// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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
template <typename BcAlgTraits>
class MomentumSymmetryElemKernel : public Kernel
{
public:
  MomentumSymmetryElemKernel(
    const stk::mesh::MetaData& metaData,
    const SolutionOptions& solnOpts,
    VectorFieldType* velocity,
    ScalarFieldType* viscosity,
    ElemDataRequests& faceDataPreReqs,
    ElemDataRequests& elemDataPreReqs);

  virtual ~MomentumSymmetryElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>& lhs,
    SharedMemView<DoubleType*>& rhs,
    ScratchViews<DoubleType>& faceScratchViews,
    ScratchViews<DoubleType>& elemScratchViews,
    int elemFaceOrdinal);

private:
  MomentumSymmetryElemKernel() = delete;

  const unsigned viscosity_;
  const unsigned velocityNp1_;
  const unsigned coordinates_;
  const unsigned exposedAreaVec_;
  const double includeDivU_;

  MasterElement* meSCS_{nullptr};
  const double penaltyFactor_;

  /// Shape functions
  AlignedViewType<
    DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]>
    vf_shape_function_{"view_face_shape_func"};
};

} // namespace nalu
} // namespace sierra

#endif /* MOMENTUMSYMMETRYELEMKERNEL_H */
