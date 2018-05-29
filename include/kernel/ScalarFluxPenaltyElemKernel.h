/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ScalarFluxPenaltyElemKernel_h
#define ScalarFluxPenaltyElemKernel_h

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

/** specificed open bc (face/elem) kernel for continuity equation (pressure DOF)
 */
template<typename BcAlgTraits>
class ScalarFluxPenaltyElemKernel: public Kernel
{
public:
  ScalarFluxPenaltyElemKernel(
    const stk::mesh::MetaData &metaData,
    const SolutionOptions &solnOpts,
    ScalarFieldType *scalarQ,
    ScalarFieldType *bcScalarQ,
    ScalarFieldType *diffFluxCoeff,
    ElemDataRequests &faceDataPreReqs,
    ElemDataRequests &elemDataPreReqs);

  virtual ~ScalarFluxPenaltyElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**> &lhs,
    SharedMemView<DoubleType*> &rhs,
    ScratchViews<DoubleType> &faceScratchViews,
    ScratchViews<DoubleType> &elemScratchViews,
    int elemFaceOrdinal);

private:
  ScalarFluxPenaltyElemKernel() = delete;

  ScalarFieldType *scalarQ_{nullptr};
  ScalarFieldType *bcScalarQ_{nullptr};
  ScalarFieldType *diffFluxCoeff_{nullptr};
  VectorFieldType *coordinates_{nullptr};
  GenericFieldType *exposedAreaVec_{nullptr};

  const double penaltyFac_;
  const bool shiftedGradOp_;
  MasterElement *meSCS_{nullptr};
  
  /// Shape functions
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_shape_function_ {"view_face_shape_func"};
};

}  // nalu
}  // sierra

#endif /* MOMENTUMSYMMETRYELEMKERNEL_H */
