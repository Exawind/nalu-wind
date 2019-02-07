/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/TurbKineticEnergySSTDESSrcElemKernel.h"
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
TurbKineticEnergySSTDESSrcElemKernel<AlgTraits>::
  TurbKineticEnergySSTDESSrcElemKernel(
    const stk::mesh::BulkData& bulkData,
    const SolutionOptions& solnOpts,
    ElemDataRequests& dataPreReqs,
    const bool lumpedMass)
  : Kernel(),
    lumpedMass_(lumpedMass),
    shiftedGradOp_(solnOpts.get_shifted_grad_op("velocity")),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    tkeProdLimitRatio_(solnOpts.get_turb_model_constant(TM_tkeProdLimitRatio)),
    cDESke_(solnOpts.get_turb_model_constant(TM_cDESke)),
    cDESkw_(solnOpts.get_turb_model_constant(TM_cDESkw)),
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
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  maxLengthScale_ = get_field_ordinal(metaData, "sst_max_length_scale");
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
  dataPreReqs.add_gathered_nodal_field(maxLengthScale_, 1);
  dataPreReqs.add_gathered_nodal_field(fOneBlend_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  if (shiftedGradOp_)
    dataPreReqs.add_master_element_call(
      SCV_SHIFTED_GRAD_OP, CURRENT_COORDINATES);
  else
    dataPreReqs.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
}

template <typename AlgTraits>
TurbKineticEnergySSTDESSrcElemKernel<
  AlgTraits>::~TurbKineticEnergySSTDESSrcElemKernel()
{
}

template <typename AlgTraits>
void
TurbKineticEnergySSTDESSrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**>& lhs,
  SharedMemView<DoubleType*>& rhs,
  ScratchViews<DoubleType>& scratchViews)
{
  NALU_ALIGNED DoubleType w_dudx[AlgTraits::nDim_][AlgTraits::nDim_];

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
  SharedMemView<DoubleType*>& v_maxLengthScale =
    scratchViews.get_scratch_view_1D(maxLengthScale_);
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
    DoubleType maxLengthScale = 0.0;
    DoubleType fOneBlend = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i)
      for (int j = 0; j < AlgTraits::nDim_; ++j)
        w_dudx[i][j] = 0.0;

    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      const DoubleType r = v_shape_function_(ip, ic);

      rho += r * v_densityNp1(ic);
      tke += r * v_tkeNp1(ic);
      sdr += r * v_sdrNp1(ic);
      tvisc += r * v_tvisc(ic);
      maxLengthScale += r * v_maxLengthScale(ic);
      fOneBlend += r * v_fOneBlend(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        const DoubleType ui = v_velocityNp1(ic, i);
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_dudx[i][j] += v_dndx(ip, ic, j) * ui;
        }
      }
    }

    DoubleType Pk = 0.0;
    for (int i = 0; i < AlgTraits::nDim_; ++i) {
      for (int j = 0; j < AlgTraits::nDim_; ++j) {
        Pk += w_dudx[i][j] * (w_dudx[i][j] + w_dudx[j][i]);
      }
    }
    Pk *= tvisc;

    // blend cDES
    const DoubleType cDES = fOneBlend * cDESkw_ + (1.0 - fOneBlend) * cDESke_;

    const DoubleType sqrtTke = stk::math::sqrt(tke);
    const DoubleType lSST = sqrtTke / betaStar_ / sdr;
    // prevent divide by zero (possible at resolved tke bcs = 0.0)
    const DoubleType lDES =
      stk::math::max(1.0e-16, stk::math::min(lSST, cDES * maxLengthScale));

    // tke factor
    const DoubleType tkeFac = rho * sqrtTke / lDES;

    // dissipation and production (limited)
    DoubleType Dk = tkeFac * tke;
    Pk = stk::math::min(Pk, tkeProdLimitRatio_ * Dk);

    // assemble RHS and LHS
    rhs(nearestNode) += (Pk - Dk) * scV;
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {
      lhs(nearestNode, ic) += v_shape_function_(ip, ic) * tkeFac * scV;
    }
  }
}

INSTANTIATE_KERNEL(TurbKineticEnergySSTDESSrcElemKernel)

} // namespace nalu
} // namespace sierra
