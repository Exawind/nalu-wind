/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumHybridTurbElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

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

template <typename AlgTraits>
MomentumHybridTurbElemKernel<AlgTraits>::MomentumHybridTurbElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  VectorFieldType*,
  ElemDataRequests& dataPreReqs)
  : shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity"))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = get_field_ordinal(metaData, "velocity");
  densityNp1_ = get_field_ordinal(metaData, "density");
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke");
  alphaNp1_ = get_field_ordinal(metaData, "adaptivity_parameter");
  mutij_ = get_field_ordinal(metaData, "tensor_turbulent_viscosity");

  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(alphaNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(
    mutij_, AlgTraits::nDim_, AlgTraits::nDim_);

  // master element data
  dataPreReqs.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCS_SHAPE_FCN, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCS_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCS_GRAD_OP, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
MomentumHybridTurbElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  NALU_ALIGNED DoubleType w_mutijScs[AlgTraits::nDim_ * AlgTraits::nDim_];

  const auto& v_uNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  const auto& v_rhoNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  const auto& v_tkeNp1 = scratchViews.get_scratch_view_1D(tkeNp1_);
  const auto& v_alphaNp1 = scratchViews.get_scratch_view_1D(alphaNp1_);
  const auto& v_mutij = scratchViews.get_scratch_view_3D(mutij_);

  const auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  const auto& v_scs_areav = meViews.scs_areav;
  const auto& v_shape_function = meViews.scs_shape_fcn;
  const auto& v_dndx = shiftedGradOp_ ? meViews.dndx_shifted : meViews.dndx;

  const int* lrscv = meSCS_->adjacentNodes();

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {

    // left and right nodes for this ip
    const int il = lrscv[2 * ip];
    const int ir = lrscv[2 * ip + 1];

    // save off some offsets
    const int ilNdim = il * AlgTraits::nDim_;
    const int irNdim = ir * AlgTraits::nDim_;

    // zero out vector that prevail over all components
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      const int offset = i * AlgTraits::nDim_;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        w_mutijScs[offset + j] = 0.0;
      }
    }
    DoubleType rhoScs = 0.0;
    DoubleType tkeScs = 0.0;
    DoubleType alphaScs = 0.0;

    // determine scs values of interest
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // save off shape function
      const DoubleType r = v_shape_function(ip, ic);

      rhoScs += r * v_rhoNp1(ic);
      tkeScs += r * v_tkeNp1(ic);
      alphaScs += r * v_alphaNp1(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        const int offset = i * AlgTraits::nDim_;
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_mutijScs[offset + j] += r * v_mutij(ic, i, j);
        }
      }
    }

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      const int icNdim = ic * AlgTraits::nDim_;

      for (int i = 0; i < AlgTraits::nDim_; ++i) {

        // tke stress term
        const DoubleType twoThirdRhoTke =
          2.0 / 3.0 * alphaScs * rhoScs * tkeScs * v_scs_areav(ip, i);

        const int indexL = ilNdim + i;
        const int indexR = irNdim + i;

        const int offseti = i * AlgTraits::nDim_;

        // Hybrid turbulence diffusion term; -(mu^jk*dui/dxk + mu^ik*duj/dxk -
        // 2/3*rho*tke*del_ij)*Aj
        DoubleType lhs_riC_i = 0.0;
        for (int j = 0; j < AlgTraits::nDim_; ++j) {

          const DoubleType axj = v_scs_areav(ip, j);
          const DoubleType uj = v_uNp1(ic, j);

          // -mut^jk*dui/dxk*A_j; fixed i over j loop; see below..
          DoubleType lhsfacDiff_i = 0.0;
          const int offsetj = j * AlgTraits::nDim_;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            lhsfacDiff_i += -w_mutijScs[offsetj + k] * v_dndx(ip, ic, k) * axj;
          }
          lhsfacDiff_i *= alphaScs;
          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;

          // -mut^ik*duj/dxk*A_j
          DoubleType lhsfacDiff_j = 0.0;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            lhsfacDiff_j += -w_mutijScs[offseti + k] * v_dndx(ip, ic, k) * axj;
          }
          lhsfacDiff_j *= alphaScs;

          // lhs; il then ir
          lhs(indexL, icNdim + j) += lhsfacDiff_j;
          lhs(indexR, icNdim + j) -= lhsfacDiff_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j * uj;
          rhs(indexR) += lhsfacDiff_j * uj;
        }

        // deal with accumulated lhs and flux for -mut^jk*dui/dxk*Aj
        lhs(indexL, icNdim + i) += lhs_riC_i;
        lhs(indexR, icNdim + i) -= lhs_riC_i;
        const DoubleType ui = v_uNp1(ic, i);
        rhs(indexL) -= lhs_riC_i * ui + twoThirdRhoTke;
        rhs(indexR) += lhs_riC_i * ui + twoThirdRhoTke;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumHybridTurbElemKernel)

} // namespace nalu
} // namespace sierra
