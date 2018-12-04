/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corp.                                           */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef RadTransWallElemKernel_h
#define RadTransWallElemKernel_h

#include "FieldTypeDef.h"
#include "kernel/Kernel.h"

#include <stk_mesh/base/BulkData.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class ElemDataRequests;
class MasterElement;
class RadiativeTransportEquationSystem;
class TimeIntegrator;

/** Add Int I sj*njds 
 */
template<typename BcAlgTraits>
class RadTransWallElemKernel: public Kernel
{
public:
  RadTransWallElemKernel(
      const stk::mesh::BulkData&,
      RadiativeTransportEquationSystem *radEqSystem,
      const bool &,
      ElemDataRequests&);

  virtual ~RadTransWallElemKernel();

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
  RadTransWallElemKernel() = delete;

  ScalarFieldType *intensity_{nullptr};
  ScalarFieldType *bcIntensity_{nullptr};
  GenericFieldType *exposedAreaVec_{nullptr};

  const RadiativeTransportEquationSystem *radEqSystem_;
  
  // Integration point to node mapping 
  const int *ipNodeMap_{nullptr};

  // scratch space
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_shape_function_{"vf_shape_function"};
  AlignedViewType<DoubleType[BcAlgTraits::nDim_]> v_Sk_{"v_Sk"};
};

}  // nalu
}  // sierra

#endif /* RadTransWallElemKernel_h */
