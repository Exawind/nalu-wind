// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumSSTAMSForcingNodeKernel.h"
#include "Realm.h"
#include "SolutionOptions.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"
#include <SimdInterface.h>

namespace sierra {
namespace nalu {

MomentumSSTAMSForcingNodeKernel::MomentumSSTAMSForcingNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPNodeKernel<MomentumSSTAMSForcingNodeKernel>(),
    betaStar_(solnOpts.get_turb_model_constant(TM_betaStar)),
    forceCl_(solnOpts.get_turb_model_constant(TM_forCl)),
    Ceta_(solnOpts.get_turb_model_constant(TM_forCeta)),
    Ct_(solnOpts.get_turb_model_constant(TM_forCt)),
    blT_(solnOpts.get_turb_model_constant(TM_forBlT)),
    blKol_(solnOpts.get_turb_model_constant(TM_forBlKol)),
    forceFactor_(solnOpts.get_turb_model_constant(TM_forFac)),
    cMu_(solnOpts.get_turb_model_constant(TM_v2cMu)),
    periodicForcingLengthX_(
      solnOpts.get_turb_model_constant(TM_periodicForcingLengthX)),
    periodicForcingLengthY_(
      solnOpts.get_turb_model_constant(TM_periodicForcingLengthY)),
    periodicForcingLengthZ_(
      solnOpts.get_turb_model_constant(TM_periodicForcingLengthZ)),
    nDim_(bulk.mesh_meta_data().spatial_dimension()),
    eastVector_("eastVector", nDim_),
    northVector_("northVector", nDim_)
{
  const auto& meta = bulk.mesh_meta_data();

  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");

  coordinatesID_ = get_field_ordinal(meta, solnOpts.get_coordinates_name());
  const std::string velField = "velocity";
  velocityID_ = get_field_ordinal(meta, velField);
  viscosityID_ = get_field_ordinal(meta, "viscosity");
  turbViscID_ = get_field_ordinal(meta, "turbulent_viscosity");
  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);
  tkeNp1ID_ = get_field_ordinal(meta, "turbulent_ke", stk::mesh::StateNP1);
  sdrNp1ID_ =
    get_field_ordinal(meta, "specific_dissipation_rate", stk::mesh::StateNP1);
  betaID_ = get_field_ordinal(meta, "k_ratio");
  MijID_ = get_field_ordinal(meta, "metric_tensor");
  minDistID_ = get_field_ordinal(meta, "minimum_distance_to_wall");

  // average quantities
  avgVelocityID_ = get_field_ordinal(meta, "average_velocity");
  avgResAdeqID_ = get_field_ordinal(meta, "avg_res_adequacy_parameter");

  // output quantities
  forcingCompID_ = get_field_ordinal(meta, "forcing_components");

  // setup vectors
  if (solnOpts.RANSBelowKs_) {
    if (!solnOpts.eastVector_.empty() && !solnOpts.northVector_.empty()) {
      DoubleView::HostMirror eastHost("eastHost", nDim_);
      DoubleView::HostMirror northHost("northHost", nDim_);
      for (int i = 0; i < nDim_; i++) {
        eastHost(i) = solnOpts.eastVector_[i];
        northHost(i) = solnOpts.northVector_[i];
      }
      Kokkos::deep_copy(eastVector_, eastHost);
      Kokkos::deep_copy(northVector_, northHost);
    } else {
      // vectors are required but unallocated
      throw std::runtime_error(
        "Using rans_below_ks requires definitions of east and north");
    }
  }
}

void
MomentumSSTAMSForcingNodeKernel::setup(Realm& realm)
{
  // Time information
  dt_ = realm.get_time_step();
  time_ = realm.get_current_time();

  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  coordinates_ = fieldMgr.get_field<double>(coordinatesID_);
  velocity_ = fieldMgr.get_field<double>(velocityID_);
  viscosity_ = fieldMgr.get_field<double>(viscosityID_);
  tvisc_ = fieldMgr.get_field<double>(turbViscID_);
  density_ = fieldMgr.get_field<double>(densityNp1ID_);
  tke_ = fieldMgr.get_field<double>(tkeNp1ID_);
  sdr_ = fieldMgr.get_field<double>(sdrNp1ID_);
  beta_ = fieldMgr.get_field<double>(betaID_);
  Mij_ = fieldMgr.get_field<double>(MijID_);
  minDist_ = fieldMgr.get_field<double>(minDistID_);
  avgVelocity_ = fieldMgr.get_field<double>(avgVelocityID_);
  avgResAdeq_ = fieldMgr.get_field<double>(avgResAdeqID_);
  forcingComp_ = fieldMgr.get_field<double>(forcingCompID_);
  RANSBelowKs_ = realm.solutionOptions_->RANSBelowKs_;
  z0_ = realm.solutionOptions_->roughnessHeight_;
}

KOKKOS_FUNCTION
void
MomentumSSTAMSForcingNodeKernel::execute(
  NodeKernelTraits::LhsType&,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  // Scratch work arrays
  NALU_ALIGNED NodeKernelTraits::DblType
    coords[NodeKernelTraits::NDimMax]; // coordinates
  NALU_ALIGNED NodeKernelTraits::DblType
    avgU[NodeKernelTraits::NDimMax]; // averageVelocity
  NALU_ALIGNED NodeKernelTraits::DblType
    fluctU[NodeKernelTraits::NDimMax]; // fluctuatingVelocity

  const NodeKernelTraits::DblType dualVolume = dualNodalVolume_.get(node, 0);

  const NodeKernelTraits::DblType mu = viscosity_.get(node, 0);
  const NodeKernelTraits::DblType tvisc = tvisc_.get(node, 0);
  const NodeKernelTraits::DblType rho = density_.get(node, 0);
  const NodeKernelTraits::DblType tke =
    stk::math::max(tke_.get(node, 0), 1.0e-12);
  const NodeKernelTraits::DblType sdr = sdr_.get(node, 0);
  const NodeKernelTraits::DblType beta = beta_.get(node, 0);
  const NodeKernelTraits::DblType wallDist = minDist_.get(node, 0);
  const NodeKernelTraits::DblType avgResAdeq = avgResAdeq_.get(node, 0);

  for (int d = 0; d < nDim_; d++) {
    avgU[d] = avgVelocity_.get(node, d);
    fluctU[d] = velocity_.get(node, d) - avgVelocity_.get(node, d);
    coords[d] = coordinates_.get(node, d);
  }

  const NodeKernelTraits::DblType eps = betaStar_ * tke * sdr;

  const NodeKernelTraits::DblType smallCl_ = 2.0;
  const NodeKernelTraits::DblType clOffset_ = 0.2;

  NodeKernelTraits::DblType length =
    (forceCl_ + (1.0 - stk::math::max(beta, 1.0 - clOffset_)) / clOffset_ *
                  (smallCl_ - forceCl_)) *
    stk::math::pow(beta * tke, 1.5) / eps;
  length = stk::math::max(
    length,
    Ceta_ * (stk::math::pow(mu / rho, 0.75) / stk::math::pow(eps, 0.25)));

  const NodeKernelTraits::DblType lengthY = stk::math::min(length, wallDist);

  NodeKernelTraits::DblType T_beta = beta * tke / eps;
  T_beta = stk::math::max(T_beta, Ct_ * stk::math::sqrt(mu / rho / eps));
  T_beta = blT_ * T_beta;

  // FIXME : Make this aware of wall direction, for now it is
  //         generalized using lengthY for all directions
  const NodeKernelTraits::DblType clipLengthX =
    stk::math::min(lengthY, periodicForcingLengthX_);
  const NodeKernelTraits::DblType clipLengthY =
    stk::math::min(lengthY, periodicForcingLengthY_);
  const NodeKernelTraits::DblType clipLengthZ =
    stk::math::min(lengthY, periodicForcingLengthZ_);

  const NodeKernelTraits::DblType ratioX =
    std::floor(periodicForcingLengthX_ / (clipLengthX + 1.e-12) + 0.5);
  const NodeKernelTraits::DblType ratioY =
    std::floor(periodicForcingLengthY_ / (clipLengthY + 1.e-12) + 0.5);
  const NodeKernelTraits::DblType ratioZ =
    std::floor(periodicForcingLengthZ_ / (clipLengthZ + 1.e-12) + 0.5);

  const NodeKernelTraits::DblType denomX = periodicForcingLengthX_ / ratioX;
  const NodeKernelTraits::DblType denomY = periodicForcingLengthY_ / ratioY;
  const NodeKernelTraits::DblType denomZ = periodicForcingLengthZ_ / ratioZ;

  const NodeKernelTraits::DblType ax = M_PI / denomX;
  const NodeKernelTraits::DblType ay = M_PI / denomY;
  const NodeKernelTraits::DblType az = M_PI / denomZ;

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
    forceFactor_ * stk::math::sqrt(beta * v2) / T_beta;

  const NodeKernelTraits::DblType prod_r_temp =
    (F_target * dt_) * (hX * fluctU[0] + hY * fluctU[1] + hZ * fluctU[2]);

  const NodeKernelTraits::DblType prod_r_sgn =
    stk::math::if_then_else(prod_r_temp < 0.0, -1.0, 1.0);
  const NodeKernelTraits::DblType prod_r_abs = prod_r_sgn * prod_r_temp;

  const NodeKernelTraits::DblType prod_r =
    stk::math::if_then_else(prod_r_abs >= 1.0e-15, prod_r_temp, 0.0);

  const NodeKernelTraits::DblType b_kol =
    stk::math::min(blKol_ * stk::math::sqrt(mu * eps / rho) / tke, 1.0);

  const NodeKernelTraits::DblType bhat = stk::math::if_then_else(
    (1.0 - b_kol) > 0.0, (1.0 - beta) / (1.0 - b_kol), 10000.0);

  NodeKernelTraits::DblType C_F_tmp =
    -1.0 * stk::math::tanh(
             1.0 - 1.0 / stk::math::sqrt(stk::math::min(avgResAdeq, 1.0)));

  C_F_tmp =
    C_F_tmp *
    (1.0 - stk::math::min(stk::math::tanh(10.0 * (bhat - 1.0)) + 1.0, 1.0));

  const NodeKernelTraits::DblType C_F =
    stk::math::if_then_else(prod_r >= 0.0, F_target * C_F_tmp, 0.0);

  // Now we determine the actual forcing field
  NodeKernelTraits::DblType gX = C_F * hX;
  NodeKernelTraits::DblType gY = C_F * hY;
  NodeKernelTraits::DblType gZ = C_F * hZ;

  if (RANSBelowKs_) {
    // relationship b/w sand grain roughness height, k_s, and aerodynamic
    // roughness, z0, as described in ref. Bau11, Eq. (2.29)
    const NodeKernelTraits::DblType k_s = 30. * z0_;
    int gravity_i;
    for (int i = 0; i < 3; ++i) {
      if ((eastVector_(i) == 0.0) && (northVector_(i) == 0.0)) {
        gravity_i = i;
      }
    }
    if (coords[gravity_i] <= k_s) {
      gX = 0.0;
      gY = 0.0;
      gZ = 0.0;
    }
  }

  forcingComp_.get(node, 0) = gX;
  forcingComp_.get(node, 1) = gY;
  forcingComp_.get(node, 2) = gZ;

  rhs(0) += dualVolume * gX;
  rhs(1) += dualVolume * gY;
  rhs(2) += dualVolume * gZ;
}

} // namespace nalu
} // namespace sierra
