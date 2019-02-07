/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SCALARUPWADVDIFFELEMKERNEL_H
#define SCALARUPWADVDIFFELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;
template <typename T> class PecletFunction;
class EquationSystem;

/** CVFEM scalar upwind advection/diffusion kernel
 */
template<typename AlgTraits>
class ScalarUpwAdvDiffElemKernel: public Kernel
{
public:
  ScalarUpwAdvDiffElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    EquationSystem*,
    ScalarFieldType*,
    VectorFieldType*,
    ScalarFieldType*,
    ElemDataRequests&);

  virtual ~ScalarUpwAdvDiffElemKernel();

  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

  virtual DoubleType van_leer(
    const DoubleType &dqm,
    const DoubleType &dqp);

private:
  ScalarUpwAdvDiffElemKernel() = delete;

  const SolutionOptions& solnOpts_;

  unsigned scalarQ_ {stk::mesh::InvalidOrdinal};
  unsigned Gjq_ {stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};

  /// Left right node indicators
  const int* lrscv_;

  /// Name of the primitive variable (for upwind options lookup in solution options)
  const std::string dofName_;

  double alpha_;
  double alphaUpw_;
  double hoUpwind_;
  bool useLimiter_;
  double om_alpha_;
  double om_alphaUpw_;
  const bool shiftedGradOp_;
  const double small_{1.0e-16};

  /// Peclet function
  PecletFunction<DoubleType>* pecletFunction_{nullptr};

  /// Shape functions
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_ { "v_shape_func" };
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_adv_shape_function_{"v_adv_shape_function"};
};

}  // nalu
}  // sierra

#endif /* SCALARUPWADVDIFFELEMKERNEL_H */
