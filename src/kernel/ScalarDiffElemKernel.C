/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ScalarDiffElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "TimeIntegrator.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<typename AlgTraits>
ScalarDiffElemKernel<AlgTraits>::ScalarDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* scalarQ,
  ScalarFieldType* diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : Kernel(),
    scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    lrscv_(sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_)->adjacentNodes()),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(scalarQ->name()))
{
  // Save of required fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(AlgTraits::topo_);
  get_scs_shape_fn_data<AlgTraits>([&](double* ptr){meSCS->shape_fcn(ptr);}, v_shape_function_);

  dataPreReqs.add_cvfem_surface_me(meSCS);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  dataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
}

template<typename AlgTraits>
ScalarDiffElemKernel<AlgTraits>::~ScalarDiffElemKernel()
{}

template<typename AlgTraits>
void
ScalarDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  SharedMemView<DoubleType*>& v_scalarQ = scratchViews.get_scratch_view_1D(scalarQ_);
  SharedMemView<DoubleType*>& v_diffFluxCoeff = scratchViews.get_scratch_view_1D(diffFluxCoeff_);

  SharedMemView<DoubleType**>& v_scs_areav = scratchViews.get_me_views(CURRENT_COORDINATES).scs_areav;
  SharedMemView<DoubleType***>& v_dndx = shiftedGradOp_
    ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_shifted 
    : scratchViews.get_me_views(CURRENT_COORDINATES).dndx;

  // start the assembly
  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv_[2*ip];
    const int ir = lrscv_[2*ip+1];

    // compute ip property
    DoubleType diffFluxCoeffIp = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function_(ip,ic);
      diffFluxCoeffIp += r*v_diffFluxCoeff(ic);
    }

    // assemble to rhs and lhs
    DoubleType qDiff = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      DoubleType lhsfacDiff = 0.0;
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        lhsfacDiff += -diffFluxCoeffIp*v_dndx(ip,ic,j)*v_scs_areav(ip,j);
      }
      qDiff += lhsfacDiff*v_scalarQ(ic);

      // lhs; il then ir
      lhs(il,ic) += lhsfacDiff;
      lhs(ir,ic) -= lhsfacDiff;
    }

    // rhs; il then ir
    rhs[il] -= qDiff;
    rhs[ir] += qDiff;
  }
}

INSTANTIATE_KERNEL(ScalarDiffElemKernel)

}  // nalu
}  // sierra
