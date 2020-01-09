// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "kernel/MomentumSSTTAMSDiffElemKernel.h"
#include "AlgTraits.h"
#include "EigenDecomposition.h"
#include "master_element/MasterElement.h"
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

#include "utils/TAMSUtils.h"
#include "ngp_utils/NgpTypes.h"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MomentumSSTTAMSDiffElemKernel<AlgTraits>::MomentumSSTTAMSDiffElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* viscosity,
  ElemDataRequests& dataPreReqs)
  : viscosity_(viscosity->mesh_meta_data_ordinal()),
    includeDivU_(solnOpts.includeDivU_),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    CMdeg_(solnOpts.get_turb_model_constant(TM_CMdeg)),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity"))
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = get_field_ordinal(metaData, "velocity");
  densityNp1_ = get_field_ordinal(metaData, "density");
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke");
  sdrNp1_ = get_field_ordinal(metaData, "specific_dissipation_rate");
  alpha_ = get_field_ordinal(metaData, "k_ratio");
  Mij_ = get_field_ordinal(metaData, "metric_tensor");

  avgVelocity_ = get_field_ordinal(metaData, "average_velocity");

  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  meSCS_ = sierra::nalu::MasterElementRepo::get_surface_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_surface_me(meSCS_);

  // fields
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(sdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(avgVelocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(alpha_, 1);
  dataPreReqs.add_gathered_nodal_field(
    Mij_, AlgTraits::nDim_, AlgTraits::nDim_);

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
MomentumSSTTAMSDiffElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const auto& v_uNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  const auto& v_viscosity = scratchViews.get_scratch_view_1D(viscosity_);
  const auto& v_rhoNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  const auto& v_tkeNp1 = scratchViews.get_scratch_view_1D(tkeNp1_);
  const auto& v_sdrNp1 = scratchViews.get_scratch_view_1D(sdrNp1_);
  const auto& v_avgU = scratchViews.get_scratch_view_2D(avgVelocity_);
  const auto& v_alpha = scratchViews.get_scratch_view_1D(alpha_);
  const auto& v_Mij = scratchViews.get_scratch_view_3D(Mij_);

  const auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  const auto& v_scs_areav = meViews.scs_areav;
  const auto& v_shape_function = meViews.scs_shape_fcn;
  const auto& v_dndx = shiftedGradOp_ ? meViews.dndx_shifted : meViews.dndx;

  const int* lrscv = meSCS_->adjacentNodes();

  // Mij, eigenvectors and eigenvalues
  DoubleType Mij[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
  DoubleType Q[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
  DoubleType D[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
  for (unsigned i = 0; i < AlgTraits::nDim_; i++)
    for (unsigned j = 0; j < AlgTraits::nDim_; j++)
      Mij[i][j] = 0.0;

  // determine scs values of interest
  for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
    for (unsigned i = 0; i < AlgTraits::nDim_; i++)
      for (unsigned j = 0; j < AlgTraits::nDim_; j++)
        Mij[i][j] += v_Mij(ic, i, j) / AlgTraits::nodesPerElement_;
  }

  EigenDecomposition::sym_diagonalize<DoubleType>(Mij, Q, D);

  // At this point we have Q, the eigenvectors and D the eigenvalues of Mij,
  // so to create M43, we use Q D^(4/3) Q^T
  DoubleType M43[nalu_ngp::NDimMax][nalu_ngp::NDimMax];
  for (unsigned i = 0; i < AlgTraits::nDim_; i++)
    for (unsigned j = 0; j < AlgTraits::nDim_; j++)
      M43[i][j] = 0.0;

  const double fourThirds = 4. / 3.;
  for (unsigned k = 0; k < AlgTraits::nDim_; k++) {
    const DoubleType D43 = stk::math::pow(D[k][k], fourThirds);
    for (unsigned i = 0; i < AlgTraits::nDim_; i++) {
      for (unsigned j = 0; j < AlgTraits::nDim_; j++) {
        M43[i][j] += Q[i][k] * Q[j][k] * D43;
      }
    }
  }

  // Compute CM43
  DoubleType CM43 =
    tams_utils::get_M43_constant<DoubleType, nalu_ngp::NDimMax>(D, CMdeg_);

  for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {

    // left and right nodes for this ip
    const int il = lrscv[2 * ip];
    const int ir = lrscv[2 * ip + 1];

    // save off some offsets
    const int ilNdim = il * AlgTraits::nDim_;
    const int irNdim = ir * AlgTraits::nDim_;

    DoubleType muScs = 0.0;
    DoubleType rhoScs = 0.0;
    DoubleType tkeScs = 0.0;
    DoubleType sdrScs = 0.0;
    DoubleType alphaScs = 0.0;
    DoubleType avgDivU = 0.0;

    // determine scs values of interest
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // save off shape function
      const DoubleType r = v_shape_function(ip, ic);

      muScs += r * v_viscosity(ic);
      rhoScs += r * v_rhoNp1(ic);
      tkeScs += r * v_tkeNp1(ic);
      sdrScs += r * v_sdrNp1(ic);
      alphaScs += r * v_alpha(ic);

      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        const DoubleType avgUj = v_avgU(ic, j);
        avgDivU += avgUj * v_dndx(ip, ic, j);
      }
    }

    // This is the divU term for the average quantities in the model for
    // tau_ij^SGRS Since we are letting SST calculate it's normal mu_t, we need
    // to scale by alpha here
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      const DoubleType avgDivUstress = 2.0 / 3.0 * alphaScs * muScs * avgDivU *
                                       v_scs_areav(ip, i) * includeDivU_;
      const int indexL = ilNdim + i;
      const int indexR = irNdim + i;
      rhs(indexL) -= avgDivUstress;
      rhs(indexR) += avgDivUstress;
    }

    const DoubleType epsilon13Scs =
      stk::math::pow(betaStar_ * tkeScs * sdrScs, 1.0 / 3.0);

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // Related to LHS
      const int icNdim = ic * AlgTraits::nDim_;

      for (int i = 0; i < AlgTraits::nDim_; ++i) {

        const int indexL = ilNdim + i;
        const int indexR = irNdim + i;

        // Hybrid turbulence diffusion term; -(mu^jk*dui/dxk + mu^ik*duj/dxk -
        // 2/3*rho*tke*del_ij)*Aj
        DoubleType lhs_riC_i = 0.0;
        DoubleType lhs_riCSGRS_i = 0.0;
        for (int j = 0; j < AlgTraits::nDim_; ++j) {

          const DoubleType axj = v_scs_areav(ip, j);
          const DoubleType fluctUj = v_uNp1(ic, j) - v_avgU(ic, j);
          const DoubleType avgUj = v_avgU(ic, j);

          // -mut^jk*dui/dxk*A_j; fixed i over j loop; see below..
          DoubleType lhsfacDiff_i = 0.0;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            lhsfacDiff_i += -rhoScs * CM43 * epsilon13Scs * M43[j][k] *
                            v_dndx(ip, ic, k) * axj;
          }

          // SGRS (average) term, scaled by alpha
          const DoubleType lhsfacDiffSGRS_i =
            -alphaScs * muScs * v_dndx(ip, ic, j) * axj;

          // lhs; il then ir
          lhs_riC_i += lhsfacDiff_i;
          lhs_riCSGRS_i += lhsfacDiffSGRS_i;

          // -mut^ik*duj/dxk*A_j
          DoubleType lhsfacDiff_j = 0.0;
          for (int k = 0; k < AlgTraits::nDim_; ++k) {
            lhsfacDiff_j += -rhoScs * CM43 * epsilon13Scs * M43[i][k] *
                            v_dndx(ip, ic, k) * axj;
          }

          // SGRS (average) term, scaled by alpha
          const DoubleType lhsfacDiffSGRS_j =
            -alphaScs * muScs * v_dndx(ip, ic, i) * axj;

          // NOTE: lhs (implicit only from the fluctuating term, u' = u - <u>,
          // so the lhs can function as normal as it will only take the
          // instantaneous part of the fluctuation u and the rhs can just stick
          // with the fluctuating quantity
          lhs(indexL, icNdim + j) += lhsfacDiff_j;
          lhs(indexR, icNdim + j) -= lhsfacDiff_j;
          // rhs; il then ir
          rhs(indexL) -= lhsfacDiff_j * fluctUj + lhsfacDiffSGRS_j * avgUj;
          rhs(indexR) += lhsfacDiff_j * fluctUj + lhsfacDiffSGRS_j * avgUj;
        }

        // lhs handled only for fluctuating term (see NOTE above)
        // deal with accumulated lhs and flux for -mut^jk*dui/dxk*Aj
        lhs(indexL, icNdim + i) += lhs_riC_i;
        lhs(indexR, icNdim + i) -= lhs_riC_i;
        const DoubleType fluctUi = v_uNp1(ic, i) - v_avgU(ic, i);
        const DoubleType avgUi = v_avgU(ic, i);

        rhs(indexL) -= lhs_riC_i * fluctUi + lhs_riCSGRS_i * avgUi;
        rhs(indexR) += lhs_riC_i * fluctUi + lhs_riCSGRS_i * avgUi;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumSSTTAMSDiffElemKernel)

} // namespace nalu
} // namespace sierra
