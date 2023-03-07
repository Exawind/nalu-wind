// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/ABLWallFluxesAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpReduceUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "wind_energy/MoninObukhov.h"
#include "wind_energy/BdyLayerStatistics.h"
#include "utils/LinearInterpolation.h"
#include "wind_energy/BdyLayerStatistics.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

namespace {

/* A function that applies Basu et al.'s algorithm 1 or 2 to compute
 * base wall friction velocity and flux.  Later in the class, that
 * information is taken and turned into a stress vector and flux that
 * can have local fluctuations.
 *
 */
template <typename PsiFunc>
KOKKOS_FUNCTION void
compute_fluxes(
  const double tol,
  const double up,
  const double Tp,
  const double zp,
  PsiFunc Psi_m_func,
  PsiFunc Psi_h_func,
  const double kappa,
  const double z0,
  const double g,
  const double Tref,
  const double Psi_m_factor,
  const double Psi_h_factor,
  const int algorithmType,
  double& frictionVelocity,
  double& temperatureFlux,
  double& Tsurface)
{

  // Set Psi_h and Psi_m initially to zero.
  double Psi_h = 0.0;
  double Psi_m = 0.0;

  // Enter the iterative solver loop and iterate until convergence
  frictionVelocity = 0.0;
  double frictionVelocityOld = 1.0E10;
  double temperatureFluxOld = 1.0E10;
  double L = 1.0E10;
  double frictionVelocityDelta = 1.0E10;
  double temperatureFluxDelta = 1.0E10;
  ;
  int iterMax = 1000;
  int iter = 0;

  while (((frictionVelocityDelta > tol) || (temperatureFluxDelta > tol)) &&
         (iter < iterMax)) {
    // Update the old values.
    frictionVelocityOld = frictionVelocity;
    temperatureFluxOld = temperatureFlux;

    // Compute friction velocity using Monin-Obukhov similarity.
    frictionVelocity = (kappa * up) / (std::log(zp / z0) - Psi_m);

    // If given surface temperature, compute heat flux using Monin-Obukhov
    // similarity.
    if (algorithmType == 2) {
      double deltaT = Tp - Tsurface;
      temperatureFlux =
        -(deltaT * frictionVelocity * kappa) / (std::log(zp / z0) - Psi_m);
    }

    // Compute Obukhov length.
    if (temperatureFlux == 0.0) {
      L = 1.0E10;
    } else {
      L =
        -(Tref * std::pow(frictionVelocity, 3)) / (kappa * g * temperatureFlux);
    }

    // Recompute Psi_h and Psi_m.
    Psi_h = Psi_h_func(zp / L, Psi_h_factor);
    Psi_m = Psi_m_func(zp / L, Psi_m_factor);

    // Compute changes in solution.
    frictionVelocityDelta = std::abs(frictionVelocity - frictionVelocityOld);
    temperatureFluxDelta = std::abs(temperatureFlux - temperatureFluxOld);

    // If given surface flux, compute the surface temperature.
    if (algorithmType == 1) {
      Tsurface =
        Tp + (temperatureFlux * ((std::log(zp / z0) - Psi_h) /
                                 (std::max(frictionVelocity, 0.001) * kappa)));
    }

    // Add to the iteration count.
    iter++;
  }
}

} // namespace

template <typename BcAlgTraits>
ABLWallFluxesAlg<BcAlgTraits>::ABLWallFluxesAlg(
  Realm& realm,
  stk::mesh::Part* part,
  WallFricVelAlgDriver& algDriver,
  const bool useShifted,
  const YAML::Node& node)
  : Algorithm(realm, part),
    algDriver_(algDriver),
    faceData_(realm.meta_data()),
    elemData_(realm.meta_data()),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    velocityNp1_(
      get_field_ordinal(realm.meta_data(), "velocity", stk::mesh::StateNP1)),
    bcVelocity_(get_field_ordinal(realm.meta_data(), "wall_velocity_bc")),
    temperatureNp1_(
      get_field_ordinal(realm.meta_data(), "temperature", stk::mesh::StateNP1)),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    bcHeatFlux_(get_field_ordinal(realm.meta_data(), "heat_flux_bc")),
    wallHeatFlux_(get_field_ordinal(
      realm.meta_data(), "wall_heat_flux_bip", realm.meta_data().side_rank())),
    specificHeat_(get_field_ordinal(realm.meta_data(), "specific_heat")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    wallShearStress_(get_field_ordinal(
      realm.meta_data(),
      "wall_shear_stress_bip",
      realm.meta_data().side_rank())),
    wallNormDist_(get_field_ordinal(
      realm.meta_data(),
      "wall_normal_distance_bip",
      realm.meta_data().side_rank())),
    useShifted_(useShifted),
    meFC_(MasterElementRepo::get_surface_master_element_on_dev(
      BcAlgTraits::FaceTraits::topo_)),
    meSCS_(MasterElementRepo::get_surface_master_element_on_dev(
      BcAlgTraits::ElemTraits::topo_))
{
  faceData_.add_cvfem_face_me(meFC_);
  elemData_.add_cvfem_surface_me(meSCS_);

  faceData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  faceData_.add_gathered_nodal_field(bcVelocity_, BcAlgTraits::nDim_);
  elemData_.add_gathered_nodal_field(temperatureNp1_, 1);
  faceData_.add_gathered_nodal_field(density_, 1);
  faceData_.add_gathered_nodal_field(bcHeatFlux_, 1);
  faceData_.add_gathered_nodal_field(specificHeat_, 1);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(wallNormDist_, BcAlgTraits::numFaceIp_);

  auto shp_fcn = useShifted_ ? FC_SHIFTED_SHAPE_FCN : FC_SHAPE_FCN;
  faceData_.add_master_element_call(shp_fcn, CURRENT_COORDINATES);

  // Load the user data from the input file.
  load(node);
}

template <typename BcAlgTraits>
void
ABLWallFluxesAlg<BcAlgTraits>::load(const YAML::Node& node)
{
  // Read in the table of surface heat flux or surface temperature versus time
  // and split out into vectors for each input quantity.
  ListArray<DblType> tableData{
    {tableTimes_[0], tableFluxes_[0], tableSurfaceTemperatures_[0],
     tableWeights_[0]},
    {tableTimes_[1], tableFluxes_[1], tableSurfaceTemperatures_[1],
     tableWeights_[1]}};
  get_if_present<ListArray<DblType>>(
    node, "surface_heating_table", tableData, tableData);
  auto nTimes = tableData.size();
  tableTimes_.resize(nTimes);
  tableFluxes_.resize(nTimes);
  tableSurfaceTemperatures_.resize(nTimes);
  tableWeights_.resize(nTimes);
  for (std::vector<DblType>::size_type i = 0; i < nTimes; i++) {
    tableTimes_[i] = tableData[i][0];
    tableFluxes_[i] = tableData[i][1];
    tableSurfaceTemperatures_[i] = tableData[i][2];
    tableWeights_[i] = tableData[i][3];
  }

  // Read in the surface roughness.
  get_if_present<DblType>(node, "roughness_height", z0_, z0_);

  // Get the gravity information.
  std::vector<double> gravity_vector;
  const int ndim = realm_.spatialDimension_;
  gravity_vector.resize(ndim);
  gravity_vector = realm_.solutionOptions_->gravity_;
  get_if_present(
    node, "gravity_vector_component", gravityVectorComponent_,
    gravityVectorComponent_);
  gravity_ = std::abs(gravity_vector[gravityVectorComponent_ - 1]);

  // Read in the reference temperature.
  get_if_present<DblType>(node, "reference_temperature", Tref_, Tref_);

  // Read in the averaging type.
  get_if_present(
    node, "monin_obukhov_averaging_type", averagingType_, averagingType_);

  // If planar averaging is used, check to make sure ABL boundary layer
  // statistics are enabled.
  if (averagingType_ == "planar") {
    if (realm_.bdyLayerStats_ == nullptr) {
      throw std::runtime_error(
        "ABL lower boundary condition with planar averaging requires ABL "
        "statistics (enable 'boundary_layer_statistics' in input file).");
    }
  }

  // Read in the fluctuation type.
  get_if_present(
    node, "fluctuation_model", fluctuationModel_, fluctuationModel_);

  // Read in what temperature to use as reference in calculating fluctuating
  // heat flux.
  get_if_present(
    node, "fluctuating_temperature_ref", fluctuatingTempRef_,
    fluctuatingTempRef_);

  // Read in M-O scaling law parameters.
  get_if_present(node, "kappa", kappa_, kappa_);
  get_if_present(node, "beta_m", beta_m_, beta_m_);
  get_if_present(node, "beta_h", beta_h_, beta_h_);
  get_if_present(node, "gamma_m", gamma_m_, gamma_m_);
  get_if_present(node, "gamma_h", gamma_h_, gamma_h_);
}

template <typename BcAlgTraits>
void
ABLWallFluxesAlg<BcAlgTraits>::execute()
{
  namespace mo = abl_monin_obukhov;
  using FaceElemSimdData =
    sierra::nalu::nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpUtau = fieldMgr.template get_field<double>(wallFricVel_);
  auto ngpqSurf = fieldMgr.template get_field<double>(wallHeatFlux_);
  auto ngptauSurf = fieldMgr.template get_field<double>(wallShearStress_);

  // Get the current time and interpolate flux/surface temperature in time.
  const double currTime = realm_.get_current_time();
  DblType currFlux;
  DblType currSurfaceTemperature;
  DblType currWeight;
  utils::linear_interp(tableTimes_, tableFluxes_, currTime, currFlux);
  utils::linear_interp(
    tableTimes_, tableSurfaceTemperatures_, currTime, currSurfaceTemperature);
  utils::linear_interp(tableTimes_, tableWeights_, currTime, currWeight);

  // Bring class members into local scope for device capture
  const unsigned velID = velocityNp1_;
  const unsigned bcVelID = bcVelocity_;
  const unsigned tempID = temperatureNp1_;
  const unsigned rhoID = density_;
  const unsigned bcHeatFluxID = bcHeatFlux_;
  const unsigned specHeatID = specificHeat_;
  const unsigned areaVecID = exposedAreaVec_;
  const unsigned wDistID = wallNormDist_;

  auto* meSCS = meSCS_;

  const DoubleType gravity = gravity_;
  const DoubleType z0 = z0_;
  const DoubleType Tref = Tref_;
  const DoubleType kappa = kappa_;
  const DoubleType beta_m = beta_m_;
  const DoubleType beta_h = beta_h_;
  const DoubleType gamma_m = gamma_m_;
  const DoubleType gamma_h = gamma_h_;

  const bool useShifted = useShifted_;

  DblType avgFactor = 0.0;
  DblType tempAverage = Tref_;
  DblType velMagAverage = 0.0;
  Kokkos::View<double[3]> velAverage("vel_average");
  auto hVelAverage = Kokkos::create_mirror_view(velAverage);

  if (averagingType_ == "planar") {
    avgFactor = 1.0;
    BdyLayerStatistics::HostArrayType h = realm_.bdyLayerStats_->abl_heights();
    realm_.bdyLayerStats_->velocity(h[1], hVelAverage.data());
    realm_.bdyLayerStats_->velocity_magnitude(h[1], &velMagAverage);
    realm_.bdyLayerStats_->temperature(h[1], &tempAverage);
  }
  Kokkos::deep_copy(velAverage, hVelAverage);

  DblType fluctuationFactor = (fluctuationModel_ != "none") ? 1.0 : 0.0;
  DblType MoengFactor = (fluctuationModel_ == "Moeng") ? 1.0 : 0.0;
  DblType fluctuatingTempRef =
    (fluctuatingTempRef_ == "reference") ? Tref_ : currSurfaceTemperature;
  const double eps = 1.0e-8;

  const stk::mesh::Selector sel =
    realm_.meta_data().locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const auto utauOps = nalu_ngp::simd_face_elem_field_updater(ngpMesh, ngpUtau);
  const auto qSurfOps =
    nalu_ngp::simd_face_elem_field_updater(ngpMesh, ngpqSurf);
  const auto tauSurfOps =
    nalu_ngp::simd_face_elem_field_updater(ngpMesh, ngptauSurf);

  // Reducer to accumulate the area-weighted utau sum as well as total area for
  // wall boundary of this specific topology.
  nalu_ngp::ArraySimdDouble2 utauSum(0.0);
  Kokkos::Sum<nalu_ngp::ArraySimdDouble2> utauReducer(utauSum);

  const std::string algName = "ABLWallFluxesAlg_" +
                              std::to_string(BcAlgTraits::faceTopo_) + "_" +
                              std::to_string(BcAlgTraits::elemTopo_);

  nalu_ngp::run_face_elem_par_reduce(
    algName, meshInfo, faceData_, elemData_, sel,
    KOKKOS_LAMBDA(
      FaceElemSimdData & feData, nalu_ngp::ArraySimdDouble2 & uSum) {
      // Unit normal vector
      NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];

      // Velocities
      NALU_ALIGNED DoubleType velIp[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType velOppNode[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType bcVelIp[BcAlgTraits::nDim_];

      // Surface stress
      NALU_ALIGNED DoubleType tauSurf_calc[BcAlgTraits::nDim_];
      DoubleType utau_calc;
      DoubleType qSurf_calc;

      auto& scrViewsFace = feData.simdFaceView;
      auto& scrViewsElem = feData.simdElemView;
      const auto& v_vel = scrViewsElem.get_scratch_view_2D(velID);
      const auto& v_bcvel = scrViewsFace.get_scratch_view_2D(bcVelID);
      const auto& v_temp = scrViewsElem.get_scratch_view_1D(tempID);
      const auto& v_rho = scrViewsFace.get_scratch_view_1D(rhoID);
      const auto& v_bcHeatFlux = scrViewsFace.get_scratch_view_1D(bcHeatFluxID);
      const auto& v_specHeat = scrViewsFace.get_scratch_view_1D(specHeatID);
      const auto& v_areavec = scrViewsFace.get_scratch_view_2D(areaVecID);
      const auto& v_wallnormdist = scrViewsFace.get_scratch_view_1D(wDistID);

      const auto meViews = scrViewsFace.get_me_views(CURRENT_COORDINATES);
      const auto& v_shape_fcn =
        useShifted ? meViews.fc_shifted_shape_fcn : meViews.fc_shape_fcn;

      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {

        const int nodeL = meSCS->opposingNodes(feData.faceOrd, ip);

        DoubleType aMag = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          aMag += v_areavec(ip, d) * v_areavec(ip, d);
        }
        aMag = stk::math::sqrt(aMag);

        // unit normal and reset velocities and stress.
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          nx[d] = v_areavec(ip, d) / aMag;
          velIp[d] = 0.0;
          velOppNode[d] = 0.0;
          bcVelIp[d] = 0.0;
          tauSurf_calc[d] = 0.0;
        }
        utau_calc = eps;
        qSurf_calc = 0.0;

        const DoubleType zh = v_wallnormdist(ip);

        // Compute quantities at the boundary integration points
        DoubleType heatFluxIp = 0.0;
        DoubleType rhoIp = 0.0;
        DoubleType CpIp = 0.0;
        DoubleType tempIp = 0.0;
        DoubleType tempOppNode = 0.0;

        for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic) {
          const DoubleType r = v_shape_fcn(ip, ic);
          heatFluxIp += r * v_bcHeatFlux(ic);
          rhoIp += r * v_rho(ic);
          CpIp += r * v_specHeat(ic);
          tempIp += r * v_temp(ic);

          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            velIp[d] += r * v_vel(ic, d);
            bcVelIp[d] += r * v_bcvel(ic, d);
          }
        }

        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          velOppNode[d] = v_vel(nodeL, d);
        }
        tempOppNode = v_temp(nodeL);

        DoubleType uIpTangential = 0.0;
        DoubleType uOppNodeTangential = 0.0;
        DoubleType uAverageTangential = 0.0;

        NALU_ALIGNED DoubleType uiIpTan[BcAlgTraits::nDim_];
        NALU_ALIGNED DoubleType uiOppNodeTan[BcAlgTraits::nDim_];
        NALU_ALIGNED DoubleType uiAverageTan[BcAlgTraits::nDim_];
        NALU_ALIGNED DoubleType uiBcTan[BcAlgTraits::nDim_];
        for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
          uiIpTan[i] = 0.0;
          uiOppNodeTan[i] = 0.0;
          uiAverageTan[i] = 0.0;
          uiBcTan[i] = 0.0;

          for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
            DoubleType ninj = nx[i] * nx[j];
            if (i == j) {
              const DoubleType om_ninj = 1.0 - ninj;
              uiIpTan[i] += om_ninj * velIp[j];
              uiOppNodeTan[i] += om_ninj * velOppNode[j];
              uiAverageTan[i] += om_ninj * velAverage[j];
              uiBcTan[i] += om_ninj * bcVelIp[j];
            } else {
              uiIpTan[i] -= ninj * velIp[j];
              uiOppNodeTan[i] -= ninj * velOppNode[j];
              uiAverageTan[i] -= ninj * velAverage[j];
              uiBcTan[i] -= ninj * bcVelIp[j];
            }
          }
          uIpTangential +=
            (uiIpTan[i] - uiBcTan[i]) * (uiIpTan[i] - uiBcTan[i]);
          uOppNodeTangential +=
            (uiOppNodeTan[i] - uiBcTan[i]) * (uiOppNodeTan[i] - uiBcTan[i]);
          uAverageTangential +=
            (uiAverageTan[i] - uiBcTan[i]) * (uiAverageTan[i] - uiBcTan[i]);
        }

        uIpTangential = stk::math::sqrt(uIpTangential);
        uOppNodeTangential = stk::math::sqrt(uOppNodeTangential);
        uAverageTangential = stk::math::sqrt(uAverageTangential);

        DoubleType SAvg = velMagAverage;
        DoubleType S = uOppNodeTangential;
        for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
          // Get variables in Moeng-like notation.
          DoubleType uiTan = uiOppNodeTan[i];
          DoubleType uiTanAvg = uiAverageTan[i];

          // Get the directionality.
          tauSurf_calc[i] =
            (1.0 - avgFactor) * (uiTan / stk::math::max(S, eps)) +
            (avgFactor) * (uiTanAvg / stk::math::max(SAvg, eps));

          // then, get the fluctuation.
          DoubleType sgnVel =
            stk::math::if_then_else(uiTanAvg >= 0.0, 1.0, -1.0);
          DoubleType term1 =
            sgnVel * stk::math::max(
                       ((1.0 - MoengFactor) * SAvg + MoengFactor * S) *
                         stk::math::abs(uiTanAvg),
                       eps);
          DoubleType term2 = SAvg * (uiTan - uiTanAvg);
          DoubleType term3 =
            sgnVel * stk::math::max((SAvg * stk::math::abs(uiTanAvg)), eps);
          tauSurf_calc[i] *=
            (1.0 - avgFactor) * 1.0 +
            (avgFactor * fluctuationFactor) * ((term1 + term2) / term3);
        }

        DoubleType thetai = tempOppNode;
        DoubleType thetaiAvg = tempAverage;
        DoubleType sgnDeltaTheta = stk::math::if_then_else(
          (thetaiAvg - fluctuatingTempRef) >= 0.0, 1.0, -1.0);
        DoubleType term1 =
          sgnDeltaTheta * stk::math::max(
                            ((1.0 - MoengFactor) * SAvg + MoengFactor * S) *
                              stk::math::abs(thetaiAvg - fluctuatingTempRef),
                            eps);
        DoubleType term2 = SAvg * (thetai - thetaiAvg);
        DoubleType term3 =
          sgnDeltaTheta *
          stk::math::max(
            (SAvg * stk::math::abs((thetaiAvg - fluctuatingTempRef))), eps);
        qSurf_calc = (1.0 - avgFactor) * 1.0 + (avgFactor * fluctuationFactor) *
                                                 ((term1 + term2) / term3);

        DoubleType u_MO =
          (1.0 - avgFactor) * uOppNodeTangential + (avgFactor)*velMagAverage;
        DoubleType temp_MO =
          (1.0 - avgFactor) * (tempOppNode) + (avgFactor) * (tempAverage);
        DoubleType q_MO = currFlux;

        const DoubleType term = stk::math::log(zh / z0);

        for (int si = 0; si < feData.numSimdElems; ++si) {

          DblType tol = 1.0E-6;
          DblType utau = 0.0;
          NALU_ALIGNED DblType tauSurf[3];
          DblType qSurf = 0.0;

          // Compute fluxes with algorithm 1.
          DblType utau_alg1 = 0.0;
          DblType Tsurf_alg1 = 0.0;
          DblType givenFlux = currFlux;
          int algType = 1;
          if (stk::simd::get_data(q_MO, si) < -eps) {
            compute_fluxes(
              tol, stk::simd::get_data(u_MO, si),
              stk::simd::get_data(temp_MO, si), stk::simd::get_data(zh, si),
              mo::psim_stable<double>, mo::psih_stable<double>,
              stk::simd::get_data(kappa, si), stk::simd::get_data(z0, si),
              stk::simd::get_data(gravity, si), stk::simd::get_data(Tref, si),
              stk::simd::get_data(beta_m, si), stk::simd::get_data(beta_h, si),
              algType, utau_alg1, givenFlux, Tsurf_alg1);
          } else if (stk::simd::get_data(q_MO, si) > eps) {
            compute_fluxes(
              tol, stk::simd::get_data(u_MO, si),
              stk::simd::get_data(temp_MO, si), stk::simd::get_data(zh, si),
              mo::psim_unstable<double>, mo::psih_unstable<double>,
              stk::simd::get_data(kappa, si), stk::simd::get_data(z0, si),
              stk::simd::get_data(gravity, si), stk::simd::get_data(Tref, si),
              stk::simd::get_data(gamma_m, si),
              stk::simd::get_data(gamma_h, si), algType, utau_alg1, givenFlux,
              Tsurf_alg1);
          } else {
            utau_alg1 = stk::simd::get_data(kappa, si) *
                        stk::simd::get_data(u_MO, si) /
                        stk::simd::get_data(term, si);
            Tsurf_alg1 = stk::simd::get_data(temp_MO, si);
          }

          // Compute fluxes with algorithm 2.
          DblType utau_alg2 = 0.0;
          DblType qSurf_alg2 = 0.0;
          DblType givenSurfaceTemperature = currSurfaceTemperature;
          algType = 2;
          if (stk::simd::get_data(temp_MO, si) - currSurfaceTemperature > eps) {
            compute_fluxes(
              tol, stk::simd::get_data(u_MO, si),
              stk::simd::get_data(temp_MO, si), stk::simd::get_data(zh, si),
              mo::psim_stable<double>, mo::psih_stable<double>,
              stk::simd::get_data(kappa, si), stk::simd::get_data(z0, si),
              stk::simd::get_data(gravity, si), stk::simd::get_data(Tref, si),
              stk::simd::get_data(beta_m, si), stk::simd::get_data(beta_h, si),
              algType, utau_alg2, qSurf_alg2, givenSurfaceTemperature);
          } else if (
            stk::simd::get_data(temp_MO, si) - currSurfaceTemperature < -eps) {
            compute_fluxes(
              tol, stk::simd::get_data(u_MO, si),
              stk::simd::get_data(temp_MO, si), stk::simd::get_data(zh, si),
              mo::psim_unstable<double>, mo::psih_unstable<double>,
              stk::simd::get_data(kappa, si), stk::simd::get_data(z0, si),
              stk::simd::get_data(gravity, si), stk::simd::get_data(Tref, si),
              stk::simd::get_data(gamma_m, si),
              stk::simd::get_data(gamma_h, si), algType, utau_alg2, qSurf_alg2,
              givenSurfaceTemperature);
          } else {
            utau_alg2 = stk::simd::get_data(kappa, si) *
                        stk::simd::get_data(u_MO, si) /
                        stk::simd::get_data(term, si);
            qSurf_alg2 = 0.0;
          }

          // Combine the fluxes computed with the two different algorithms based
          // on the user-selected weighting.
          utau = (1.0 - (currWeight - 1.0)) * utau_alg1 +
                 (currWeight - 1.0) * utau_alg2;
          qSurf = ((1.0 - (currWeight - 1.0)) * currFlux +
                   (currWeight - 1.0) * qSurf_alg2) *
                  stk::simd::get_data(rhoIp, si) *
                  stk::simd::get_data(CpIp, si);

          // Compute the fluctuating flux fields.
          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            tauSurf[d] = -stk::simd::get_data(rhoIp, si) * utau * utau *
                         stk::simd::get_data(tauSurf_calc[d], si);
          }
          qSurf *= stk::simd::get_data(qSurf_calc, si);

          // Collect the data back up to put on the fields.
          stk::simd::set_data(utau_calc, si, utau);
          stk::simd::set_data(qSurf_calc, si, qSurf);
          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            stk::simd::set_data(tauSurf_calc[d], si, tauSurf[d]);
          }
        }
        utauOps(feData, ip) = utau_calc;
        qSurfOps(feData, ip) = qSurf_calc;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          tauSurfOps(feData, ip * BcAlgTraits::nDim_ + d) = tauSurf_calc[d];
        }

        // Accumulate utau for statistics output
        uSum.array_[0] += utau_calc * aMag;
        uSum.array_[1] += aMag;
      }
    },
    utauReducer);

  algDriver_.accumulate_utau_area_sum(utauSum.array_[0], utauSum.array_[1]);
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ABLWallFluxesAlg)

} // namespace nalu
} // namespace sierra
