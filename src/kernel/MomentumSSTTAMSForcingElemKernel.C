// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernel/MomentumSSTTAMSForcingElemKernel.h"
#include "AlgTraits.h"
#include "EigenDecomposition.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"
#include "ngp_utils/NgpTypes.h"

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
MomentumSSTTAMSForcingElemKernel<AlgTraits>::MomentumSSTTAMSForcingElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ScalarFieldType* viscosity,
  ScalarFieldType* turbViscosity,
  ElemDataRequests& dataPreReqs)
  : viscosity_(viscosity->mesh_meta_data_ordinal()),
    turbViscosity_(turbViscosity->mesh_meta_data_ordinal()),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    cMu_(solnOpts.get_turb_model_constant(TM_v2cMu)),
    forceCl_(solnOpts.get_turb_model_constant(TM_forCl)),
    Ceta_(solnOpts.get_turb_model_constant(TM_forCeta)),
    Ct_(solnOpts.get_turb_model_constant(TM_forCt)),
    blT_(solnOpts.get_turb_model_constant(TM_forBlT)),
    blKol_(solnOpts.get_turb_model_constant(TM_forBlKol)),
    forceFactor_(solnOpts.get_turb_model_constant(TM_forFac))
{
  pi_ = stk::math::acos(-1.0);
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  velocityNp1_ = get_field_ordinal(metaData, "velocity");
  densityNp1_ = get_field_ordinal(metaData, "density");
  tkeNp1_ = get_field_ordinal(metaData, "turbulent_ke");

  sdrNp1_ = get_field_ordinal(metaData, "specific_dissipation_rate");
  alpha_ = get_field_ordinal(metaData, "k_ratio");
  Mij_ = get_field_ordinal(metaData, "metric_tensor");

  avgVelocity_ = get_field_ordinal(metaData, "average_velocity");
  avgTime_ = get_field_ordinal(metaData, "rans_time_scale");

  avgResAdeq_ = get_field_ordinal(metaData, "avg_res_adequacy_parameter");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  minDist_ = get_field_ordinal(metaData, "minimum_distance_to_wall");

  meSCV_ = MasterElementRepo::get_volume_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV_);

  // fields
  dataPreReqs.add_coordinates_field(
    coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(viscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(turbViscosity_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(tkeNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(sdrNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(avgVelocity_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(alpha_, 1);
  dataPreReqs.add_gathered_nodal_field(avgTime_, 1);
  dataPreReqs.add_gathered_nodal_field(minDist_, 1);
  dataPreReqs.add_gathered_nodal_field(avgResAdeq_, 1);
  dataPreReqs.add_gathered_nodal_field(
    Mij_, AlgTraits::nDim_, AlgTraits::nDim_);

  // master element data
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_SHAPE_FCN, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
MomentumSSTTAMSForcingElemKernel<AlgTraits>::setup(
  const TimeIntegrator& timeIntegrator)
{
  time_ = timeIntegrator.get_current_time();
  dt_ = timeIntegrator.get_time_step();
}

template <typename AlgTraits>
void
MomentumSSTTAMSForcingElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& /*lhs*/,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  NALU_ALIGNED DoubleType w_coordScv[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_avgUScv[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_fluctUScv[AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_MijElem[AlgTraits::nDim_][AlgTraits::nDim_];

  const auto& v_coordinates = scratchViews.get_scratch_view_2D(coordinates_);
  const auto& v_uNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  const auto& v_viscosity = scratchViews.get_scratch_view_1D(viscosity_);
  const auto& v_turbViscosity =
    scratchViews.get_scratch_view_1D(turbViscosity_);
  const auto& v_rhoNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  const auto& v_tkeNp1 = scratchViews.get_scratch_view_1D(tkeNp1_);
  const auto& v_sdrNp1 = scratchViews.get_scratch_view_1D(sdrNp1_);
  const auto& v_avgU = scratchViews.get_scratch_view_2D(avgVelocity_);
  const auto& v_alpha = scratchViews.get_scratch_view_1D(alpha_);
  const auto& v_avgTime = scratchViews.get_scratch_view_1D(avgTime_);
  const auto& v_minDist = scratchViews.get_scratch_view_1D(minDist_);
  const auto& v_avgResAdeq = scratchViews.get_scratch_view_1D(avgResAdeq_);
  const auto& v_Mij = scratchViews.get_scratch_view_3D(Mij_);

  auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  const auto& v_scv_volume = meViews.scv_volume;
  auto& v_shape_function = meViews.scv_shape_fcn;
  const auto* ipNodeMap = meSCV_->ipNodeMap();

  for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {

    // nearest node for this ip
    const int nearestNode = ipNodeMap[ip];

    // zero out values of interest for this ip
    for (int j = 0; j < AlgTraits::nDim_; ++j) {
      w_coordScv[j] = 0.0;
      w_avgUScv[j] = 0.0;
      w_fluctUScv[j] = 0.0;
      for (int k = 0; k < AlgTraits::nDim_; ++k) {
        w_MijElem[j][k] = 0.0;
      }
    }

    DoubleType muScv = 0.0;
    DoubleType mu_tScv = 0.0;
    DoubleType rhoScv = 0.0;
    DoubleType tkeScv = 0.0;
    DoubleType sdrScv = 0.0;
    DoubleType avgTimeScv = 0.0;
    DoubleType alphaScv = 0.0;
    DoubleType wallDistScv = 0.0;
    DoubleType avgResAdeqScv = 0.0;

    // determine scs values of interest
    for (int ic = 0; ic < AlgTraits::nodesPerElement_; ++ic) {

      // save off shape function
      const DoubleType r = v_shape_function(ip, ic);

      muScv += r * v_viscosity(ic);
      mu_tScv += r * v_turbViscosity(ic);
      rhoScv += r * v_rhoNp1(ic);
      tkeScv += r * v_tkeNp1(ic);
      sdrScv += r * v_sdrNp1(ic);
      avgTimeScv += r * v_avgTime(ic);
      alphaScv += r * v_alpha(ic);
      wallDistScv += r * v_minDist(ic);
      avgResAdeqScv += r * v_avgResAdeq(ic);

      for (int i = 0; i < AlgTraits::nDim_; ++i) {
        w_coordScv[i] += r * v_coordinates(ic, i);
        w_avgUScv[i] += r * v_avgU(ic, i);
        w_fluctUScv[i] += r * (v_uNp1(ic, i) - v_avgU(ic, i));

        // Don't allow Mij to vary over the element, so take it as a direct
        // average over nodes
        for (int j = 0; j < AlgTraits::nDim_; ++j) {
          w_MijElem[i][j] += v_Mij(ic, i, j) / AlgTraits::nodesPerElement_;
        }
      }
    }

    const DoubleType epsScv = betaStar_ * tkeScv * sdrScv;

    // First we calculate the a_i's
    NALU_ALIGNED const DoubleType periodicForcingLength[3] = {pi_, 0.25,
                                                              3.0 / 8.0 * pi_};

    DoubleType length =
      forceCl_ * stk::math::pow(alphaScv * tkeScv, 1.5) / epsScv;
    length = stk::math::max(
      length, Ceta_ * (stk::math::pow(muScv / rhoScv, 0.75) /
                       stk::math::pow(epsScv, 0.25)));
    length = stk::math::min(length, wallDistScv);

    DoubleType T_alpha = alphaScv * tkeScv / epsScv;
    T_alpha =
      stk::math::max(T_alpha, Ct_ * stk::math::sqrt(muScv / rhoScv / epsScv));
    T_alpha = blT_ * T_alpha;

    NALU_ALIGNED DoubleType ceilLength[AlgTraits::nDim_];
    for (int d = 0; d < AlgTraits::nDim_; d++)
      ceilLength[d] = stk::math::max(length, 2.0 * w_MijElem[d][d]);

    NALU_ALIGNED DoubleType clipLength[AlgTraits::nDim_];
    for (int d = 0; d < AlgTraits::nDim_; d++)
      clipLength[d] = stk::math::min(ceilLength[d], periodicForcingLength[d]);

    // FIXME: Hack to do a round/floor/ceil/mod operation since it isnt in
    // stk::math right now
    NALU_ALIGNED DoubleType ratio[AlgTraits::nDim_];
    for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
      for (int d = 0; d < AlgTraits::nDim_; d++) {
        double tmpD = stk::simd::get_data(clipLength[d], simdIndex);
        double tmpN = stk::simd::get_data(periodicForcingLength[d], simdIndex);
        double tmp = std::floor(tmpN / tmpD + 0.5);
        stk::simd::set_data(ratio[d], simdIndex, tmp);
      }
    }

    NALU_ALIGNED DoubleType denom[AlgTraits::nDim_];
    for (int d = 0; d < AlgTraits::nDim_; d++)
      denom[d] = periodicForcingLength[d] / ratio[d];

    NALU_ALIGNED DoubleType a[AlgTraits::nDim_];
    for (int d = 0; d < AlgTraits::nDim_; d++)
      a[d] = pi_ / denom[d];

    // Then we calculate the arguments for the Taylor-Green Vortex
    NALU_ALIGNED DoubleType tgarg[nalu_ngp::NDimMax] = {0.0, 0.0, 0.0};
    for (int d = 0; d < AlgTraits::nDim_; d++)
      tgarg[d] = a[d] * (w_coordScv[d] + w_avgUScv[d] * time_);

    // Now we calculate the initial Taylor-Green field
    NALU_ALIGNED const DoubleType h[nalu_ngp::NDimMax] = {
      1.0 / 3.0 * stk::math::cos(tgarg[0]) * stk::math::sin(tgarg[1]) *
        stk::math::sin(tgarg[2]),
      -1.0 * stk::math::sin(tgarg[0]) * stk::math::cos(tgarg[1]) *
        stk::math::sin(tgarg[2]),
      2.0 / 3.0 * stk::math::sin(tgarg[0]) * stk::math::sin(tgarg[1]) *
        stk::math::cos(tgarg[2])};

    // Now we calculate the scaling of the initial field
    const DoubleType v2Scv = mu_tScv * betaStar_ * sdrScv / (cMu_ * rhoScv);
    const DoubleType F_target =
      forceFactor_ * stk::math::sqrt(alphaScv * v2Scv) / T_alpha;

    DoubleType prod_r_temp = 0.0;
    for (int d = 0; d < AlgTraits::nDim_; d++)
      prod_r_temp += h[d] * w_fluctUScv[d];
    prod_r_temp *= (F_target * dt_);

    const DoubleType prod_r_sgn =
      stk::math::if_then_else(prod_r_temp < 0.0, -1.0, 1.0);
    const DoubleType prod_r_abs = prod_r_sgn * prod_r_temp;

    const DoubleType prod_r =
      stk::math::if_then_else(prod_r_abs >= 1.0e-15, prod_r_temp, 0.0);

    const DoubleType arg1 = stk::math::sqrt(avgResAdeqScv) - 1.0;
    const DoubleType arg = stk::math::if_then_else(
      arg1 < 0.0, 1.0 - 1.0 / stk::math::sqrt(avgResAdeqScv), arg1);

    const DoubleType a_sign = stk::math::tanh(arg);

    const DoubleType a_kol = stk::math::min(
      blKol_ * stk::math::sqrt(muScv * epsScv / rhoScv) / tkeScv, 1.0);

    const DoubleType Sa = stk::math::if_then_else(
      (a_sign < 0.0),
      stk::math::if_then_else(
        (alphaScv <= a_kol), a_sign - (1.0 + a_kol - alphaScv) * a_sign,
        a_sign),
      stk::math::if_then_else(
        (alphaScv >= 1.0), a_sign - alphaScv * a_sign, a_sign));

    const DoubleType C_F = stk::math::if_then_else(
      ((avgResAdeqScv < 1.0) && (prod_r >= 0.0)), -1.0 * F_target * Sa, 0.0);

    // Now we determine the actual forcing field
    NALU_ALIGNED const DoubleType g[nalu_ngp::NDimMax] = {
      C_F * h[0], C_F * h[1], C_F * h[2]};

    // TODO: Assess viability of first approach where we don't solve a poisson
    // problem and allow the field be divergent, which should get projected out
    // anyway. This means we only have a contribution to the RHS here
    const DoubleType scV = v_scv_volume(ip);
    const int nnNdim = nearestNode * AlgTraits::nDim_;

    rhs(nnNdim + 0) += g[0] * scV;
    rhs(nnNdim + 1) += g[1] * scV;
    rhs(nnNdim + 2) += g[2] * scV;
  }
}

INSTANTIATE_KERNEL(MomentumSSTTAMSForcingElemKernel)

} // namespace nalu
} // namespace sierra
