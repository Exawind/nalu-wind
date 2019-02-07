/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/SpecificDissipationRateSSTSrcElemKernel.h"
#include "FieldTypeDef.h"
#include "SolutionOptions.h"

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
SpecificDissipationRateSSTSrcElemKernel<AlgTraits>::
  SpecificDissipationRateSSTSrcElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs,
    const bool lumpedMass)
  : Kernel(),
    lumpedMass_(lumpedMass),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity")),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    sigmaWTwo_(solnOpts.get_turb_model_constant(TM_sigmaWTwo)),
    betaOne_(solnOpts.get_turb_model_constant(TM_betaOne)),
    betaTwo_(solnOpts.get_turb_model_constant(TM_betaTwo)),
    gammaOne_(solnOpts.get_turb_model_constant(TM_gammaOne)),
    gammaTwo_(solnOpts.get_turb_model_constant(TM_gammaTwo)),
    tkeProdLimitRatio_(solnOpts.get_turb_model_constant(TM_tkeProdLimitRatio)),
    ipNodeMap_(sierra::nalu::MasterElementRepo::get_volume_master_element(
                 AlgTraits::topo_)
                 ->ipNodeMap())
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke", stk::mesh::StateNP1);
  sdrNp1_ = get_field_ordinal(metaData, "specific_dissipation_rate", stk::mesh::StateNP1);
  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  velocityNp1_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateNP1);
  tvisc_ = get_field_ordinal(metaData, "turbulent_viscosity");
  fOneBlend_ = get_field_ordinal(metaData, "sst_f_one_blending");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  MasterElement* meSCV =
    sierra::nalu::MasterElementRepo::get_volume_master_element(
      AlgTraits::topo_);

  // compute shape function
  if (lumpedMass_)
    get_scv_shape_fn_data<AlgTraits>(
      [&](double* ptr) { meSCV->shifted_shape_fcn(ptr); }, v_shape_function_);
  else
    get_scv_shape_fn_data<AlgTraits>(
      [&](double* ptr) { meSCV->shape_fcn(ptr); }, v_shape_function_);

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV);

  // fields and data
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(sdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(tvisc_, 1);
  dataPreReqs.add_gathered_nodal_field(fOneBlend_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCV_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
}

template <typename AlgTraits>
SpecificDissipationRateSSTSrcElemKernel<
  AlgTraits>::~SpecificDissipationRateSSTSrcElemKernel()
{
}

template <typename AlgTraits>
void
SpecificDissipationRateSSTSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dudx[AlgTraits::nDim_][AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dkdx[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_dwdx[AlgTraits::nDim_];

  SharedMemView<DoubleType*>& v_tkeNp1 =
    scratchViews.get_scratch_view_1D(tkeNp1_);
  SharedMemView<DoubleType*>& v_sdrNp1 =
    scratchViews.get_scratch_view_1D(sdrNp1_);
  SharedMemView<DoubleType*>& v_densityNp1 =
    scratchViews.get_scratch_view_1D(densityNp1_);
  SharedMemView<DoubleType**>& v_velocityNp1 =
    scratchViews.get_scratch_view_2D(velocityNp1_);
  SharedMemView<DoubleType*>& v_tvisc =
    scratchViews.get_scratch_view_1D(tvisc_);
  SharedMemView<DoubleType*>& v_fOneBlend =
    scratchViews.get_scratch_view_1D(fOneBlend_);
  SharedMemView<DoubleType***>& v_dndx =
    shiftedGradOp_
      ? scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv_shifted
      : scratchViews.get_me_views(CURRENT_COORDINATES).dndx_scv;
  SharedMemView<DoubleType*>& v_scv_volume =
    scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {

    // nearest node to ip
    const int nearestNode = ipNodeMap_[ip];

    // save off scvol
    const DoubleType scV = v_scv_volume(ip);

    DoubleType rho = 0.0;
    DoubleType tke = 0.0;
    DoubleType sdr = 0.0;
    DoubleType tvisc = 0.0;
    DoubleType fOneBlend = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      w_dkdx[i] = 0.0;
      w_dwdx[i] = 0.0;
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        w_dudx[i][j] = 0.0;
      }
    }

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      const DoubleType r = v_shape_function_(ip, ic);

      rho += r * v_densityNp1(ic);
      tke += r * v_tkeNp1(ic);
      sdr += r * v_sdrNp1(ic);
      tvisc += r * v_tvisc(ic);
      fOneBlend += r * v_fOneBlend(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        const DoubleType dni = v_dndx(ip, ic, i);
        const DoubleType ui = v_velocityNp1(ic, i);
        w_dkdx[i] += dni * v_tkeNp1(ic);
        w_dwdx[i] += dni * v_sdrNp1(ic);
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_dudx[i][j] += v_dndx(ip, ic, j) * ui;
        }
      }
    }

    DoubleType Pk = 0.0;
    DoubleType crossDiff = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      crossDiff += w_dkdx[i] * w_dwdx[i];
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        Pk += w_dudx[i][j] * (w_dudx[i][j] + w_dudx[j][i]);
      }
    }
    Pk *= tvisc;

    // dissipation and production (limited)
    DoubleType Dk = betaStar_ * rho * sdr * tke;
    Pk = stk::math::min(Pk, tkeProdLimitRatio_ * Dk);

    // start the blending and constants
    const DoubleType om_fOneBlend = 1.0 - fOneBlend;
    const DoubleType beta = fOneBlend * betaOne_ + om_fOneBlend * betaTwo_;
    const DoubleType gamma = fOneBlend * gammaOne_ + om_fOneBlend * gammaTwo_;
    const DoubleType sigmaD = 2.0 * om_fOneBlend * sigmaWTwo_;

    // Pw includes 1/tvisc scaling; tvisc may be zero at a dirichlet low Re
    // approach (clip)
    const DoubleType Pw = gamma * rho * Pk / stk::math::max(tvisc, 1.0e-16);
    const DoubleType Dw = beta * rho * sdr * sdr;
    const DoubleType Sw = sigmaD * rho * crossDiff / sdr;

    // assemble RHS and LHS
    rhs(nearestNode) += (Pw - Dw + Sw) * scV;
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      lhs(nearestNode, ic) +=
        v_shape_function_(ip, ic) *
        (2.0 * beta * rho * sdr + stk::math::max(Sw / sdr, 0.0)) * scV;
    }
  }
}

INSTANTIATE_KERNEL(SpecificDissipationRateSSTSrcElemKernel)

} // namespace nalu
} // namespace sierra
