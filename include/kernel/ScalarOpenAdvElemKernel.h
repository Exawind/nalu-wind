/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ScalarOpenAdvElemKernel_h
#define ScalarOpenAdvElemKernel_h

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

/** Symmetry kernel for scalar equation
 */
template<typename BcAlgTraits>
class ScalarOpenAdvElemKernel: public Kernel
{
public:
  ScalarOpenAdvElemKernel(
    const stk::mesh::MetaData &metaData,
    const SolutionOptions &solnOpts,
    EquationSystem* eqSystem,  
    ScalarFieldType *scalarQ,
    ScalarFieldType *bcScalarQ,
    VectorFieldType *Gjq,
    ScalarFieldType *diffFluxCoeff,
    ElemDataRequests &faceDataPreReqs,
    ElemDataRequests &elemDataPreReqs);

  virtual ~ScalarOpenAdvElemKernel();

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
  ScalarOpenAdvElemKernel() = delete;

  unsigned scalarQ_ {stk::mesh::InvalidOrdinal};
  unsigned bcScalarQ_ {stk::mesh::InvalidOrdinal};
  unsigned Gjq_ {stk::mesh::InvalidOrdinal};
  unsigned diffFluxCoeff_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
  
  // numerical parameters
  const double alphaUpw_;
  const double om_alphaUpw_;
  const double hoUpwind_;
  const double small_{1.0e-16};

  // Integration point to node mapping and master element for interior
  const int *faceIpNodeMap_{nullptr};
  MasterElement *meSCS_{nullptr};

  // Peclet function
  PecletFunction<DoubleType> *pecletFunction_{nullptr};

  /// Shape functions
  AlignedViewType<DoubleType[BcAlgTraits::numFaceIp_][BcAlgTraits::nodesPerFace_]> vf_adv_shape_function_ {"vf_adv_shape_function"};
};

}  // nalu
}  // sierra

#endif /* ScalarOpenAdvElemKernel_h */
