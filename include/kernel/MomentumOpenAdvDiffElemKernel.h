/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MomentumOpenAdvDiffElemKernel_h
#define MomentumOpenAdvDiffElemKernel_h

#include "master_element/MasterElement.h"

// scratch space
#include "ScratchViews.h"

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class EquationSystem;
class MasterElement;
template <typename T> class PecletFunction;
class SolutionOptions;

/** Open adv/diff kernel for momentum equation (velocity DOF)
 */
template<typename BcAlgTraits>
class MomentumOpenAdvDiffElemKernel: public Kernel
{
public:
  MomentumOpenAdvDiffElemKernel(
    const stk::mesh::MetaData &metaData,
    const SolutionOptions &solnOpts,
    EquationSystem* eqSystem,
    VectorFieldType *velocity,
    GenericFieldType *Gjui,
    ScalarFieldType *viscosity,
    ElemDataRequests &faceDataPreReqs,
    ElemDataRequests &elemDataPreReqs);

  virtual ~MomentumOpenAdvDiffElemKernel();

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
  MomentumOpenAdvDiffElemKernel() = delete;

  unsigned viscosity_ {stk::mesh::InvalidOrdinal};
  unsigned Gjui_ {stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
  unsigned velocityBc_ {stk::mesh::InvalidOrdinal};
  
  // numerical parameters
  const double alphaUpw_;
  const double om_alphaUpw_;
  const double hoUpwind_;
  const double nfEntrain_;
  const double om_nfEntrain_;
  const double includeDivU_;
  const double nocFac_;
  const bool shiftedGradOp_;
  const double small_{1.0e-16};

  // Integration point to node mapping and master element for interior
  const int *faceIpNodeMap_{nullptr};
  MasterElement *meSCS_{nullptr};

  // Peclet function
  PecletFunction<DoubleType> *pecletFunction_{nullptr};

  /// Shape functions
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_shape_function_ {"vf_shape_func"};
  AlignedViewType<DoubleType[BcAlgTraits::numScsIp_][BcAlgTraits::nodesPerElement_]> v_shape_function_ {"v_shape_func"};
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_adv_shape_function_ {"vf_adv_shape_function"};
  AlignedViewType<DoubleType[BcAlgTraits::numScsIp_][BcAlgTraits::nodesPerElement_]> v_adv_shape_function_ {"v_adv_shape_func"};
};

}  // nalu
}  // sierra

#endif /* MomentumOpenAdvDiffElemKernel_h */
