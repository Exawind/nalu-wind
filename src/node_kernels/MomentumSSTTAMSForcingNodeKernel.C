/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "node_kernels/MomentumSSTTAMSForcingNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"

#include "stk_mesh/base/MetaData.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

MomentumSSTTAMSForcingNodeKernel::MomentumSSTTAMSForcingNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPNodeKernel<MomentumSSTTAMSForcingNodeKernel>(),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    forceFactor_(solnOpts.get_turb_model_constant(TM_forFac)),
    cMu_(solnOpts.get_turb_model_constant(TM_v2cMu)),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();
  pi_ = stk::math::acos(-1.0);

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");

  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
  const std::string velField =
    solnOpts.does_mesh_move() ? "velocity_rtm" : "velocity";
  velocityRTMID_ = get_field_ordinal(meta, velField);
  viscosityID_ = get_field_ordinal(meta, "viscosity");
  turbViscID_ = get_field_ordinal(meta, "turbulent_viscosity");
  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  tkeNp1ID_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  sdrNp1ID_ =
    get_field_ordinal(meta, "specific_dissipation_rate", stk::mesh::StateNP1);
  alphaID_ = get_field_ordinal(meta, "k_ratio");
  MijID_ = get_field_ordinal(meta, "metric_tensor");
  minDistID_ = get_field_ordinal(meta, "minimum_distance_to_wall");

  // average quantities
  avgVelocityID_ = get_field_ordinal(meta, "average_velocity");
  avgDensityID_ = get_field_ordinal(meta, "average_density");
  avgResAdeqID_ = get_field_ordinal(meta, "avg_res_adequacy_parameter");
}

void
MomentumSSTTAMSForcingNodeKernel::setup(Realm& realm)
{
  // Time information
  // There's no reason for these to be SIMD datatypes... right?
  dt_ = realm.get_time_step();
  time_ = realm.get_current_time();

  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  velocity_ = fieldMgr.get_field<double>(velocityRTMID_);
  viscosity_ = fieldMgr.get_field<double>(viscosityID_);
  tvisc_ = fieldMgr.get_field<double>(turbViscID_);
  density_ = fieldMgr.get_field<double>(densityNp1ID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  alpha_ = fieldMgr.get_field<double>(alphaID_);
  Mij_ = fieldMgr.get_field<double>(MijID_);
  minDist_ = fieldMgr.get_field<double>(minDistID_);
  avgVelocity_ = fieldMgr.get_field<double>(avgVelocityID_);
  avgDensity_ = fieldMgr.get_field<double>(avgDensityID_);
  avgResAdeq_ = fieldMgr.get_field<double>(avgResAdeqID_);
}

void
MomentumSSTTAMSForcingNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  // Scratch work arrays
  NALU_ALIGNED NodeKernelTraits::DblType coords[NodeKernelTraits::NDimMax]; // coordinates
  NALU_ALIGNED NodeKernelTraits::DblType avgU[NodeKernelTraits::NDimMax];   // averageVelocity
  NALU_ALIGNED NodeKernelTraits::DblType fluctU[NodeKernelTraits::NDimMax]; // fluctuatingVelocity

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  const NodeKernelTraits::DblType mu = viscosity_.get(node, 0);
  const NodeKernelTraits::DblType tvisc = tvisc_.get(node, 0);
  const NodeKernelTraits::DblType rho = density_.get(node, 0);
  const NodeKernelTraits::DblType tke = stk::math::max(tke_.get(node, 0), 1.0e-12);
  const NodeKernelTraits::DblType sdr = sdr_.get(node, 0);
  const NodeKernelTraits::DblType alpha = alpha_.get(node, 0);
  const NodeKernelTraits::DblType wallDist = minDist_.get(node, 0);
  const NodeKernelTraits::DblType avgRho = avgDensity_.get(node, 0);
  const NodeKernelTraits::DblType avgResAdeq = avgResAdeq_.get(node, 0);

  for (int d = 0; d < nDim_; d++) {
    avgU[d] = avgVelocity_.get(node, d);
    fluctU[d] = velocity_.get(node, d) - avgVelocity_.get(node, d);
    coords[d] = coordinates_.get(node, d);
  }

  const NodeKernelTraits::DblType eps = betaStar_ * tke * sdr;

  // First we calculate the a_i's
  const double FORCING_CL = 4.0;
  const double Ceta = 70.0;
  const double Ct = 6.0;
  const double BL_T = 1.0;
  const double BL_KOL = 1.0;
  const double FORCING_FACTOR = forceFactor_;

  const NodeKernelTraits::DblType periodicForcingLengthX = pi_;
  const NodeKernelTraits::DblType periodicForcingLengthY = 0.25;
  const NodeKernelTraits::DblType periodicForcingLengthZ = 3.0 / 8.0 * pi_;

  NodeKernelTraits::DblType length =
    FORCING_CL * stk::math::pow(alpha * tke, 1.5) / eps;
  length = stk::math::max(
    length, Ceta * (stk::math::pow(mu/rho, 0.75) / stk::math::pow(eps, 0.25)));
  length = stk::math::min(length, wallDist);

  NodeKernelTraits::DblType T_alpha = alpha * tke / eps;
  T_alpha = stk::math::max(T_alpha, Ct * stk::math::sqrt(mu / rho / eps));
  T_alpha = BL_T * T_alpha;

  const NodeKernelTraits::DblType Mij_00 = Mij_.get(node, 0);
  const NodeKernelTraits::DblType Mij_11 = Mij_.get(node, 4);
  const NodeKernelTraits::DblType Mij_22 = Mij_.get(node, 8);
  const NodeKernelTraits::DblType ceilLengthX =
    stk::math::max(length, 2.0 * Mij_00);
  const NodeKernelTraits::DblType ceilLengthY =
    stk::math::max(length, 2.0 * Mij_11);
  const NodeKernelTraits::DblType ceilLengthZ =
    stk::math::max(length, 2.0 * Mij_22);

  const NodeKernelTraits::DblType clipLengthX =
    stk::math::min(ceilLengthX, periodicForcingLengthX);
  const NodeKernelTraits::DblType clipLengthY =
    stk::math::min(ceilLengthY, periodicForcingLengthY);
  const NodeKernelTraits::DblType clipLengthZ =
    stk::math::min(ceilLengthZ, periodicForcingLengthZ);

  // FIXME: Hack to do a round/floor/ceil/mod operation since it isnt in
  // stk::math right now
  NodeKernelTraits::DblType ratioX;
  NodeKernelTraits::DblType ratioY;
  NodeKernelTraits::DblType ratioZ;
  for (int simdIndex = 0; simdIndex < stk::simd::ndoubles; ++simdIndex) {
    double tmpD = stk::simd::get_data(clipLengthX, simdIndex);
    double tmpN = stk::simd::get_data(periodicForcingLengthX, simdIndex);
    double tmp = std::floor(tmpN / tmpD + 0.5);
    stk::simd::set_data(ratioX, simdIndex, tmp);

    tmpD = stk::simd::get_data(clipLengthY, simdIndex);
    tmpN = stk::simd::get_data(periodicForcingLengthY, simdIndex);
    tmp = std::floor(tmpN / tmpD + 0.5);
    stk::simd::set_data(ratioY, simdIndex, tmp);

    tmpD = stk::simd::get_data(clipLengthZ, simdIndex);
    tmpN = stk::simd::get_data(periodicForcingLengthZ, simdIndex);
    tmp = std::floor(tmpN / tmpD + 0.5);
    stk::simd::set_data(ratioZ, simdIndex, tmp);
  }

  const NodeKernelTraits::DblType denomX = periodicForcingLengthX / ratioX;
  const NodeKernelTraits::DblType denomY = periodicForcingLengthY / ratioY;
  const NodeKernelTraits::DblType denomZ = periodicForcingLengthZ / ratioZ;

  const NodeKernelTraits::DblType ax = pi_ / denomX;
  const NodeKernelTraits::DblType ay = pi_ / denomY;
  const NodeKernelTraits::DblType az = pi_ / denomZ;

  // Then we calculate the arguments for the Taylor-Green Vortex
  const NodeKernelTraits::DblType xarg = ax * (coords[0] + avgU[0] * time_);
  const NodeKernelTraits::DblType yarg = ay * (coords[1] + avgU[1] * time_);
  const NodeKernelTraits::DblType zarg = az * (coords[2] + avgU[2] * time_);

  // Now we calculate the initial Taylor-Green field
  NodeKernelTraits::DblType hX = 1. / 3. * stk::math::cos(xarg) *
                                 stk::math::sin(yarg) * stk::math::sin(zarg);
  NodeKernelTraits::DblType hY =
    -1. * stk::math::sin(xarg) * stk::math::cos(yarg) * stk::math::sin(zarg);
  NodeKernelTraits::DblType hZ = 2. / 3. * stk::math::sin(xarg) *
                                 stk::math::sin(yarg) * stk::math::cos(zarg);

  // Now we calculate the scaling of the initial field
  const NodeKernelTraits::DblType v2 = tvisc * betaStar_ * sdr / (cMu_ * rho);
  const NodeKernelTraits::DblType F_target =
    FORCING_FACTOR * stk::math::sqrt(alpha * v2) / T_alpha;

  const NodeKernelTraits::DblType prod_r_temp = (F_target*dt_)*(hX*fluctU[0] + hY*fluctU[1] + hZ*fluctU[2]);

  const NodeKernelTraits::DblType prod_r_sgn = stk::math::if_then_else(prod_r_temp < 0.0, -1.0, 1.0);
  const NodeKernelTraits::DblType prod_r_abs = prod_r_sgn * prod_r_temp;

  const NodeKernelTraits::DblType prod_r = stk::math::if_then_else(prod_r_abs >= 1.0e-15, prod_r_temp, 0.0);

  const NodeKernelTraits::DblType arg1 = stk::math::sqrt(avgResAdeq) - 1.0;
  const NodeKernelTraits::DblType arg = stk::math::if_then_else(
    arg1 < 0.0, 1.0 - 1.0 / stk::math::sqrt(avgResAdeq), arg1);

  const NodeKernelTraits::DblType a_sign = stk::math::tanh(arg);

  const NodeKernelTraits::DblType a_kol =
    stk::math::min(BL_KOL * stk::math::sqrt(mu * eps / rho) / tke, 1.0);

  const NodeKernelTraits::DblType Sa = stk::math::if_then_else(
      (a_sign < 0.0),
      stk::math::if_then_else(
        (alpha <= a_kol), a_sign - (1.0 + a_kol - alpha) * a_sign,
        a_sign),
      stk::math::if_then_else(
        (alpha >= 1.0), a_sign - alpha * a_sign, a_sign));

  const NodeKernelTraits::DblType C_F = stk::math::if_then_else(
      ((avgResAdeq < 1.0) && (prod_r >= 0.0)), -1.0 * F_target * Sa, 0.0);

  // Now we determine the actual forcing field
  NodeKernelTraits::DblType gX = C_F * hX;
  NodeKernelTraits::DblType gY = C_F * hY;
  NodeKernelTraits::DblType gZ = C_F * hZ;

  // TODO: Assess viability of first approach where we don't solve a poisson
  // problem and allow the field be divergent, which should get projected out
  // anyway. This means we only have a contribution to the RHS here

  rhs(0) += dualVolume * gX;
  rhs(1) += dualVolume * gY;
  rhs(2) += dualVolume * gZ;
}

} // namespace nalu
} // namespace sierra
