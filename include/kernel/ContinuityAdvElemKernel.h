/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef CONTINUITYADVELEMKERNEL_H
#define CONTINUITYADVELEMKERNEL_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class TimeIntegrator;
class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** CMM (BDF2) for continuity equation (pressure DOF)
 */
template<typename AlgTraits>
class ContinuityAdvElemKernel: public Kernel
{
public:
  ContinuityAdvElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  virtual ~ContinuityAdvElemKernel();

  /** Perform pre-timestep work for the computational kernel
   */
  virtual void setup(const TimeIntegrator&);

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  using Kernel::execute;
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  ContinuityAdvElemKernel() = delete;

  // extract fields; nodal
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned Gpdx_ {stk::mesh::InvalidOrdinal};
  unsigned pressure_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned Udiag_ {stk::mesh::InvalidOrdinal};

  double projTimeScale_{1.0};

  const bool meshMotion_;
  const bool shiftMdot_;
  const bool shiftPoisson_;
  const bool reducedSensitivities_;
  const double interpTogether_;
  const double om_interpTogether_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_ { "view_shape_func" };

  const int* lrscv_;
};

}  // nalu
}  // sierra

#endif /* CONTINUITYADVELEMKERNEL_H */
