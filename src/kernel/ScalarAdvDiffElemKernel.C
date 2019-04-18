/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ScalarAdvDiffElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
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
ScalarAdvDiffElemKernel<AlgTraits>::ScalarAdvDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* scalarQ,
  ScalarFieldType* diffFluxCoeff,
  ElemDataRequests& dataPreReqs)
  : scalarQ_(scalarQ->mesh_meta_data_ordinal()),
    diffFluxCoeff_(diffFluxCoeff->mesh_meta_data_ordinal()),
    shiftedGradOp_(solnOpts.get_shifted_grad_op(scalarQ->name())),
    skewSymmetric_(solnOpts.get_skew_symmetric(scalarQ->name()))
{
  // Save off required fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  massFlowRate_ = get_field_ordinal(metaData, "mass_flow_rate_scs", stk::topology::ELEM_RANK);

  meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();

  dataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(scalarQ_, 1);
  dataPreReqs.add_gathered_nodal_field(diffFluxCoeff_, 1);
  dataPreReqs.add_element_field(massFlowRate_, AlgTraits::numScsIp_);
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  if ( shiftedGradOp_ )
    dataPreReqs.add_master_element_call(SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);

  dataPreReqs.add_master_element_call(SCS_SHAPE_FCN, CURRENT_COORDINATES);

  if (skewSymmetric_)
    dataPreReqs.add_master_element_call(SCS_SHIFTED_SHAPE_FCN, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
ScalarAdvDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  auto& v_scalarQ = scratchViews.get_scratch_view_1D(scalarQ_);
  auto& v_diffFluxCoeff = scratchViews.get_scratch_view_1D(diffFluxCoeff_);
  auto& v_mdot = scratchViews.get_scratch_view_1D(massFlowRate_);

  auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  auto& v_scs_areav = meViews.scs_areav;
  auto& v_dndx = shiftedGradOp_ ? meViews.dndx_shifted : meViews.dndx;
  auto& v_shape_function = meViews.scs_shape_fcn;
  auto& v_adv_shape_function = skewSymmetric_ ? meViews.scs_shifted_shape_fcn : meViews.scs_shape_fcn;

  const int* lrscv = meSCS_->adjacentNodes();

  // start the assembly
  for ( int ip = 0; ip < AlgTraits::numScsIp_; ++ip ) {

    // left and right nodes for this ip
    const int il = lrscv[2*ip];
    const int ir = lrscv[2*ip+1];

    // save off mdot
    const DoubleType tmdot = v_mdot(ip);

    // compute ip property and
    DoubleType diffFluxCoeffIp = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {
      const DoubleType r = v_shape_function(ip,ic);
      diffFluxCoeffIp += r*v_diffFluxCoeff(ic);
    }

    // advection and diffusion
    DoubleType qAdv = 0.0;
    DoubleType qDiff = 0.0;
    for ( int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic ) {

      // advection
      const DoubleType lhsfacAdv = v_adv_shape_function(ip,ic)*tmdot;
      qAdv += lhsfacAdv*v_scalarQ(ic);

      // diffusion
      DoubleType lhsfacDiff = 0.0;
      for ( int j = 0; j < AlgTraits::nDim_; ++j ) {
        lhsfacDiff += -diffFluxCoeffIp*v_dndx(ip,ic,j)*v_scs_areav(ip,j);
      }
      qDiff += lhsfacDiff*v_scalarQ(ic);

      // lhs; il then ir
      lhs(il,ic) += lhsfacAdv + lhsfacDiff;
      lhs(ir,ic) -= lhsfacAdv + lhsfacDiff;
    }

    // rhs; il then ir
    rhs(il) -= qAdv + qDiff;
    rhs(ir) += qAdv + qDiff;
  }
}

INSTANTIATE_KERNEL(ScalarAdvDiffElemKernel)

}  // nalu
}  // sierra
