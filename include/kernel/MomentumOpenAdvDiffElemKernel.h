// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MomentumOpenAdvDiffElemKernel_h
#define MomentumOpenAdvDiffElemKernel_h

#include "Enums.h"
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
template <typename T>
class PecletFunction;
class SolutionOptions;

/** Open adv/diff kernel for momentum equation (velocity DOF)
 */
template <typename BcAlgTraits>
class MomentumOpenAdvDiffElemKernel : public Kernel
{
public:
  MomentumOpenAdvDiffElemKernel(
    const stk::mesh::MetaData& metaData,
    const SolutionOptions& solnOpts,
    EquationSystem* eqSystem,
    VectorFieldType* velocity,
    TensorFieldType* Gjui,
    ScalarFieldType* viscosity,
    ElemDataRequests& faceDataPreReqs,
    ElemDataRequests& elemDataPreReqs,
    EntrainmentMethod method = EntrainmentMethod::COMPUTED);

  virtual ~MomentumOpenAdvDiffElemKernel();

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
  MomentumOpenAdvDiffElemKernel() = delete;

  unsigned viscosity_{stk::mesh::InvalidOrdinal};
  unsigned Gjui_{stk::mesh::InvalidOrdinal};
  unsigned velocityNp1_{stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_{stk::mesh::InvalidOrdinal};
  unsigned coordinates_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned openMassFlowRate_{stk::mesh::InvalidOrdinal};
  unsigned velocityBc_{stk::mesh::InvalidOrdinal};

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
  const EntrainmentMethod entrain_{EntrainmentMethod::COMPUTED};

  // Integration point to node mapping and master element for interior
  const int* faceIpNodeMap_{nullptr};
  MasterElement* meSCS_{nullptr};
  MasterElement* meSCS_dev_{nullptr};

  // Peclet function
  PecletFunction<DoubleType>* pecletFunction_{nullptr};

  /// Shape functions
  AlignedViewType<
    DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]>
    vf_shape_function_{"vf_shape_func"};
  AlignedViewType<
    DoubleType[BcAlgTraits::numScsIp_][BcAlgTraits::nodesPerElement_]>
    v_shape_function_{"v_shape_func"};
  AlignedViewType<
    DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]>
    vf_adv_shape_function_{"vf_adv_shape_function"};
  AlignedViewType<
    DoubleType[BcAlgTraits::numScsIp_][BcAlgTraits::nodesPerElement_]>
    v_adv_shape_function_{"v_adv_shape_func"};
};

} // namespace nalu
} // namespace sierra

#endif /* MomentumOpenAdvDiffElemKernel_h */
