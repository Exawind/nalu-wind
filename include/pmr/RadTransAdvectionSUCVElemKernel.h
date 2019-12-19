// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef RadTransAdvectionSUCVElemKernel_H
#define RadTransAdvectionSUCVElemKernel_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;
class RadiativeTransportEquationSystem;
class SolutionOptions;
class TimeIntegrator;

/** Add sj*dI/dxj + SUCV 
 */
template<typename AlgTraits>
class RadTransAdvectionSUCVElemKernel: public Kernel
{
public:
  RadTransAdvectionSUCVElemKernel(
      const stk::mesh::BulkData&,
      RadiativeTransportEquationSystem *radEqSystem,
      const double sucvFac,
      const SolutionOptions&,
      ElemDataRequests&);

  virtual ~RadTransAdvectionSUCVElemKernel();

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
  RadTransAdvectionSUCVElemKernel() = delete;

  VectorFieldType *coordinates_{nullptr};
  ScalarFieldType *intensity_{nullptr};
  ScalarFieldType *absorption_{nullptr};
  ScalarFieldType *scattering_{nullptr};
  ScalarFieldType *scalarFlux_{nullptr};
  ScalarFieldType *radiationSource_{nullptr};
  ScalarFieldType *dualNodalVolume_{nullptr};

  const RadiativeTransportEquationSystem *radEqSystem_;
  
  // activate and deactivate SUCV
  const double sucvFac_;

  // edge-legth for SUCV tau
  const bool useEdgeH_;

  // 1/(4.0*pi)
  const double invFourPi_;

  const int* lrscv_;

  // scratch space
  AlignedViewType<DoubleType[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]> v_shape_function_{"v_shape_function"};
  AlignedViewType<DoubleType[AlgTraits::nDim_]> v_Sk_{"v_Sk"};
};

}  // nalu
}  // sierra

#endif /* RadTransAdvectionSUCVElemKernel_H */
