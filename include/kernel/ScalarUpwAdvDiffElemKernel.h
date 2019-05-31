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
class ScalarUpwAdvDiffElemKernel: public NGPKernel<ScalarUpwAdvDiffElemKernel<AlgTraits>>
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

  KOKKOS_FUNCTION ScalarUpwAdvDiffElemKernel() = default;

  KOKKOS_FUNCTION virtual ~ScalarUpwAdvDiffElemKernel() = default;

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  KOKKOS_FUNCTION virtual void execute(
    SharedMemView<DoubleType**, DeviceShmem>&,
    SharedMemView<DoubleType*, DeviceShmem>&,
    ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>&);

private:
  KOKKOS_FUNCTION
  DoubleType van_leer(const DoubleType &dqm, const DoubleType &dqp);

  unsigned scalarQ_ {stk::mesh::InvalidOrdinal};
  unsigned Gjq_ {stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};

  double alpha_;
  double alphaUpw_;
  double hoUpwind_;
  bool useLimiter_;
  double om_alpha_;
  double om_alphaUpw_;
  const bool shiftedGradOp_;
  const bool skewSymmetric_;
  const double small_{1.0e-16};

  //! Device pointer to the Peclet function
  PecletFunction<DoubleType>* pecletFunction_{nullptr};

  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra

#endif /* SCALARUPWADVDIFFELEMKERNEL_H */
