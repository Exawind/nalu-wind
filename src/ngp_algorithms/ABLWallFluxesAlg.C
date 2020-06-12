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

template <typename PsiFunc>
KOKKOS_FUNCTION
double calc_utau(
  const double uh,
  const double zh,
  const double term1,
  const double Tflux,
  const double Lfac,
  const double kappa,
  PsiFunc psi_func,
  const double psi_fac)
{
  const double eps = 1.0e-8;
  const double convTol = 1.0e-7;
  const double perturb = 1.0e-3;
  const int maxIters = 40;

  // Return utau as epsilon if the velocity at surface is zero
  if (stk::math::abs(uh) < eps)
    return eps;

  double utau0, utau1;
  // Proceed with normal computations
  utau0 = kappa * uh / term1;
  if (Tflux > 0.0) utau0 *= 3.0;

  utau1 = (1.0 + perturb) * utau0;

  bool converged = false;
  double utau = utau0;
  for (int k=0; k < maxIters; ++k) {
    double L0 = utau0 * utau0 * utau0 * Lfac;
    double L1 = utau1 * utau1 * utau1 * Lfac;

    double sgnq = (Tflux > 0.0) ? 1.0 : -1.0;
    L0 = - sgnq * stk::math::max(1.0e-10, stk::math::abs(L0));
    L1 = - sgnq * stk::math::max(1.1e-10, stk::math::abs(L1));

    const double znorm0 = zh / L0;
    const double znorm1 = zh / L1;

    const double denom0 = term1 - psi_func(znorm0, psi_fac);
    const double denom1 = term1 - psi_func(znorm1, psi_fac);

    const double f0 = utau0 - uh * kappa / denom0;
    const double f1 = utau1 - uh * kappa / denom1;

    double dutau = utau1 - utau0;
    dutau = (dutau > 0.0)
      ? (stk::math::max(1.0e-15, dutau))
      : (stk::math::min(-1.0e-15, dutau));

    double fprime = (f1 - f0) / dutau;
    fprime = (fprime > 0.0)
      ? (stk::math::max(1.0e-15, fprime))
      : (stk::math::min(-1.0e-15, fprime));

    const double utau_tmp = utau1;
    utau1 = utau0 - f0 / fprime;
    utau0 = utau_tmp;

    if (stk::math::abs(f1) < convTol) {
      converged = true;
      utau = stk::math::max(0.0, utau1);
      break;
    }
  }

  if (!converged) printf("Issue with utau");
  return utau;
}


//--------------------------------------------------------------------------
//-------- compute_fluxes_given_surface_temperature ------------------------
//--------------------------------------------------------------------------
template <typename PsiFunc>
KOKKOS_FUNCTION
std::vector<double> compute_fluxes_given_surface_temperature
(
    const double tol, 
    const double up, 
    const double Tp, 
    const double Tsurface, 
    const double zp, 
    PsiFunc Psi_m_func,
    PsiFunc Psi_h_func,
    const double kappa,
    const double z0,
    const double g,
    const double Tref,
    const double Psi_m_factor,
    const double Psi_h_factor    
)
{
  // This is algorithm 2 outlined by Basu et al.

  // Set Psi_h and Psi_m initially to zero.
  double Psi_h = 0.0;
  double Psi_m = 0.0;

  // Enter the iterative solver loop and iterate until convergence
  double frictionVelocity = 0.0;
  double temperatureFlux = 0.0;
  double frictionVelocityOld = 1.0E10;
  double temperatureFluxOld = 1.0E10;
  double L = 1.0E10;
  double frictionVelocityDelta = std::abs(frictionVelocity - frictionVelocityOld);
  double temperatureFluxDelta = std::abs(temperatureFlux - temperatureFluxOld);
  int iterMax = 1000;
  int iter = 0;
  while (((frictionVelocityDelta > tol ) || (temperatureFluxDelta > tol)) && (iter < iterMax))
  {
    // Update the old values.
    frictionVelocityOld = frictionVelocity;
    temperatureFluxOld = temperatureFlux;

    // Compute friction velocity using Monin-Obukhov similarity.
    frictionVelocity = (kappa * up) / (std::log(zp / z0) - Psi_m);

    // Compute heat flux using Monin-Obukhov similarity.
    double deltaT = Tp - Tsurface;
    temperatureFlux = -(deltaT * frictionVelocity * kappa) / (std::log(zp / z0) - Psi_m);

    // Compute Obukhov length.
    if (temperatureFlux == 0.0)
    {
      L = 1.0E10;
    }
    else
    {
      L = -(Tref * std::pow(frictionVelocity,3))/(kappa * g * temperatureFlux);
    }

    // Recompute Psi_H and Psi_M.
    Psi_h = Psi_h_func(zp/L, Psi_h_factor);
    Psi_m = Psi_m_func(zp/L, Psi_m_factor);

    // Compute changes in solution.
    frictionVelocityDelta = std::abs(frictionVelocity - frictionVelocityOld);
    temperatureFluxDelta = std::abs(temperatureFlux - temperatureFluxOld);

    // Add to the iteration count.
    iter++;
  
  /*  
    std::cout << "     - iteration: " << iter << std::endl;
    std::cout << "     - friction velocity = " << frictionVelocity << " " << frictionVelocityDelta << std::endl;
    std::cout << "     - temperature flux = " << temperatureFlux << " " << temperatureFluxDelta << std::endl;
    std::cout << "     - L = " << L << std::endl;
    std::cout << "     - Psi_M = " << Psi_M << std::endl;
    std::cout << "     - Psi_H = " << Psi_H << std::endl;
  */
  }
  
  std::vector<double> fluxes = {frictionVelocity,temperatureFlux};
  //utau = frictionVelocity;
  //qsurf = rho * Cp * temperatureFlux;

  return fluxes;
}



}

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
      realm.meta_data(),
      "wall_heat_flux_bip",
      realm.meta_data().side_rank())),
    specificHeat_(get_field_ordinal(realm.meta_data(), "specific_heat")),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    wallNormDist_(get_field_ordinal(
      realm.meta_data(),
      "wall_normal_distance_bip",
      realm.meta_data().side_rank())),
    useShifted_(useShifted),
    meFC_(MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
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

template<typename BcAlgTraits>
void ABLWallFluxesAlg<BcAlgTraits>::load(const YAML::Node& node)
{
  // Read in the table of surface heat flux or surface temperature versus time and split out
  // into vectors for each input quantity.
  ListArray<DblType> tableData{{tableTimes_[0],tableFluxes_[0],tableSurfaceTemperatures_[0],tableWeights_[0]},
                               {tableTimes_[1],tableFluxes_[1],tableSurfaceTemperatures_[1],tableWeights_[1]}};
  get_if_present<ListArray<DblType>>(node,"surface_heating_table",tableData,tableData);
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
  std::cout << "nTimes = " << nTimes << std::endl;
  std::cout << "Time (s)     Surface Flux (K-m/s)     Surface Temperature (K)     Blending Factor" << std::endl;
  for (std::vector<DblType>::size_type i = 0; i < nTimes; i++) {
    std::cout << tableTimes_[i] << " " << tableFluxes_[i] << " " << tableSurfaceTemperatures_[i] << " " << tableWeights_[i] << std::endl;
  }
  
  // Read in the surface roughness.
  get_if_present<DblType>(node,"roughness_height",z0_,z0_);
  std::cout << "Surface Roughness: " << z0_ << " m" << std::endl;

  // Get the gravity information.
  std::vector<double> gravity_vector;
  const int ndim = realm_.spatialDimension_;
  gravity_vector.resize(ndim);
  gravity_vector = realm_.solutionOptions_->gravity_;
  get_if_present(node,"gravity_vector_component",gravityVectorComponent_,gravityVectorComponent_);
  gravity_ = std::abs(gravity_vector[gravityVectorComponent_ - 1]);
  std::cout << "Gravity Vector: " << gravity_vector[0] << " " << gravity_vector[1] << " " << gravity_vector[2] << " m/s^2" << std::endl;
  std::cout << "Gravity: " << gravity_ << " m/s^2" << std::endl;

  // Read in the reference temperature.
  get_if_present<DblType>(node,"reference_temperature",Tref_,Tref_);
  std::cout << "Reference Temperature: " << Tref_ << " K" << std::endl;

  // Read in the averaging type.
  get_if_present(node, "monin_obukhov_averaging_type", averagingType_, averagingType_);
  std::cout << "Averaging Type: " << averagingType_ << std::endl;

  // If planar averaging is used, check to make sure ABL boundary layer statistics are enabled.
  if (averagingType_ == "planar")
  {
     if (realm_.bdyLayerStats_ == nullptr)
     {
        throw std::runtime_error("ABL lower boundary condition with planar averaging requires ABL statistics (enable 'boundary_layer_statistics' in input file).");
     }
  }

  // Read in M-O scaling law parameters.
  get_if_present(node,"kappa",kappa_,kappa_);
  get_if_present(node,"beta_m",beta_m_,beta_m_);
  get_if_present(node,"beta_h",beta_h_,beta_h_);
  get_if_present(node,"gamma_m",gamma_m_,gamma_m_);
  get_if_present(node,"gamma_h",gamma_h_,gamma_h_);
  std::cout << "kappa: " << kappa_ << std::endl;
  std::cout << "beta_m: " << beta_m_ << std::endl;
  std::cout << "beta_h: " << beta_h_ << std::endl;
  std::cout << "gamma_m: " << gamma_m_ << std::endl;
  std::cout << "gamma_h: " << gamma_h_ << std::endl;

}

template<typename BcAlgTraits>
void ABLWallFluxesAlg<BcAlgTraits>::execute()
{
  namespace mo = abl_monin_obukhov;
  using FaceElemSimdData = sierra::nalu::nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpUtau = fieldMgr.template get_field<double>(wallFricVel_);

  // Get the current time and interpolate flux/surface temperature in time.
  const double currTime = realm_.get_current_time();
  DblType currFlux;
  DblType currSurfaceTemperature;
  DblType currWeight;
  utils::linear_interp(tableTimes_, tableFluxes_, currTime, currFlux);
  utils::linear_interp(tableTimes_, tableSurfaceTemperatures_, currTime, currSurfaceTemperature);
  utils::linear_interp(tableTimes_, tableWeights_, currTime, currWeight);
  std::cout << "Flux = " << currFlux << std::endl;
  std::cout << "Surface Temperature = " << currSurfaceTemperature << std::endl;
  std::cout << "Weight = " << currWeight << std::endl;

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
  auto* meFC = meFC_;

  const DoubleType gravity = gravity_;
  const DoubleType z0 = z0_;
  const DoubleType Tref = Tref_;
  const DoubleType kappa = kappa_;
  const DoubleType beta_m = beta_m_;
  const DoubleType beta_h = beta_h_;
  const DoubleType gamma_m = gamma_m_;
  const DoubleType gamma_h = gamma_h_;

  const bool useShifted = useShifted_;

  DblType hPlanar;
  DoubleType hPlanarSIMD;
  std::vector<DblType> velPlanar(2,0.0);
  DblType tempPlanar;
  if (averagingType_ == "planar")
  {
    BdyLayerStatistics::HostArrayType h = realm_.bdyLayerStats_->abl_heights();
    realm_.bdyLayerStats_->velocity(h[1], velPlanar.data());  
    realm_.bdyLayerStats_->temperature(h[1], &tempPlanar);  
    hPlanar = h[1];
    hPlanarSIMD = h[1];
    std::cout << "h = " << h[0] << " " << h[1] << std::endl;
    std::cout << "hPlanar = " << hPlanar << std::endl;
    std::cout << "hPlanarSIMD = " << hPlanarSIMD << std::endl;
    std::cout << "velPlanar = " << velPlanar[0] << " " << velPlanar[1] << " " << velPlanar[2] << std::endl;
    std::cout << "tempPlanar = " << tempPlanar << std::endl;
  }

  NALU_ALIGNED DoubleType vv[BcAlgTraits::nDim_];
  vv = velPlanar;
  

  const double eps = 1.0e-8;
  const DoubleType Lmax = 1.0e8;

  const stk::mesh::Selector sel = realm_.meta_data().locally_owned_part()
    & stk::mesh::selectUnion(partVec_);

  const auto utauOps = nalu_ngp::simd_face_elem_field_updater(
    ngpMesh, ngpUtau);

  // Reducer to accumulate the area-weighted utau sum as well as total area for
  // wall boundary of this specific topology.
  nalu_ngp::ArraySimdDouble2 utauSum(0.0);
  Kokkos::Sum<nalu_ngp::ArraySimdDouble2> utauReducer(utauSum);

  const std::string algName = "ABLWallFluxesAlg_" +
    std::to_string(BcAlgTraits::faceTopo_) + "_" +
    std::to_string(BcAlgTraits::elemTopo_);



  nalu_ngp::run_face_elem_par_reduce(
    algName, meshInfo, faceData_, elemData_, sel,
    KOKKOS_LAMBDA(FaceElemSimdData& feData, nalu_ngp::ArraySimdDouble2& uSum) {

      // Unit normal vector
      NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];

      // Velocities
      NALU_ALIGNED DoubleType velIp[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType velOppNode[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType bcVelIp[BcAlgTraits::nDim_];
      NALU_ALIGNED DoubleType velPlanar[BcAlgTraits::nDim_];

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
      const auto& v_shape_fcn = useShifted
        ? meViews.fc_shifted_shape_fcn : meViews.fc_shape_fcn;

    //std::vector<DblType> velPlanar(BcAlgTraits::nDim_,0.0);
    //DblType tempPlanar;
    //std::vector<DoubleType> velPlanar(BcAlgTraits::nDim_,0.0);

      const int* faceIpNodeMap = meFC->ipNodeMap();
      for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {

        const int nodeR = meSCS->ipNodeMap(feData.faceOrd)[ip];
        const int nodeL = meSCS->opposingNodes(feData.faceOrd, ip);

        DoubleType aMag = 0.0;
        for (int d=0; d < BcAlgTraits::nDim_; ++d) {
          aMag += v_areavec(ip, d) * v_areavec(ip, d);
        }
        aMag = stk::math::sqrt(aMag);

        // unit normal and reset velocities
        for (int d=0; d < BcAlgTraits::nDim_; ++d) {
          nx[d] = v_areavec(ip, d) / aMag;
          velIp[d] = 0.0;
          velOppNode[d] = 0.0;
          bcVelIp[d] = 0.0;
        }

        const DoubleType zh = v_wallnormdist(ip);


        // Compute quantities at the boundary integration points
        DoubleType heatFluxIp = 0.0;
        DoubleType rhoIp = 0.0;
        DoubleType CpIp = 0.0;
        DoubleType tempIp = 0.0;
        DoubleType tempOppNode = 0.0;
        DoubleType tempPlanar = 0.0;

        for (int ic =0; ic < BcAlgTraits::nodesPerFace_; ++ic) {
          const DoubleType r = v_shape_fcn(ip, ic);
          heatFluxIp += r * v_bcHeatFlux(ic);
          rhoIp += r * v_rho(ic);
          CpIp += r * v_specHeat(ic);
          tempIp += r*v_temp(ic);

          for (int d=0; d < BcAlgTraits::nDim_; ++d) {
            velIp[d] += r * v_vel(ic, d);
            bcVelIp[d] += r * v_bcvel(ic, d);
          }
        }

        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            velOppNode[d] = v_vel(nodeL,d);
        }
        tempOppNode = v_temp(nodeL);
        std::cout << "velIp = (" << velIp[0] << " " << velIp[1] << " " << velIp[2] << "), tempIp = " << tempIp << std::endl;
        std::cout << "velOppNode = (" << velOppNode[0] << " " << velOppNode[1] << " " << velOppNode[2] << "), tempOppNode = " << tempOppNode << std::endl;

        DoubleType uTangential = 0.0;
        DoubleType uOppNodeTangential = 0.0;
        for (int i=0; i < BcAlgTraits::nDim_; ++i) {
          DoubleType uiTan = 0.0;
          DoubleType uiOppNodeTan = 0.0;
          DoubleType uiBcTan = 0.0;

          for (int j=0; j < BcAlgTraits::nDim_; ++j) {
            DoubleType ninj = nx[i] * nx[j];
            if (i == j) {
              const DoubleType om_ninj = 1.0 - ninj;
              uiTan += om_ninj * velIp[j];
              uiOppNodeTan += om_ninj * velOppNode[j];
              uiBcTan += om_ninj * bcVelIp[j];
            } else {
              uiTan -= ninj * velIp[j];
              uiOppNodeTan -= ninj * velOppNode[j];
              uiBcTan -= ninj * bcVelIp[j];
            }
          }
          uTangential += (uiTan - uiBcTan) * (uiTan - uiBcTan);
          uOppNodeTangential += (uiOppNodeTan - uiBcTan) * (uiOppNodeTan - uiBcTan);
        }
        uTangential = stk::math::sqrt(uTangential);
        uOppNodeTangential = stk::math::sqrt(uOppNodeTangential);

        const DoubleType Tflux = heatFluxIp / (rhoIp * CpIp);
        const DoubleType Lfac = stk::math::if_then_else(
          (stk::math::abs(Tflux) < eps), Lmax,
          (-Tref / (kappa * gravity * Tflux)));
        const DoubleType term = stk::math::log(zh / z0);

        DoubleType utau_calc = eps;
        std::cout << "numSimdElems = " << feData.numSimdElems << std::endl;
        for (int si = 0; si < feData.numSimdElems; ++si) {

        // Get planar averaged velocity.
      //if ((averagingType_ == "planar"))
      //{
      //  DblType velPlanar
      //  realm_.bdyLayerStats_->velocity(zh, velPlanar.data());
      //  realm_.bdyLayerStats_->temperature(zh, &tempPlanar);
     //  std::cout << zPlanar << " " << velPlanar[0] << " " << velPlanar[1] << " " << velPlanar[2] << " " << tempPlanar << std::endl;
      //}
        
/*
          std::vector<DblType> mean_fluxes_given_surf_temp(2,0.0);
          DblType tol = 1.0E-6;
          mean_fluxes_given_surf_temp = compute_fluxes_given_surface_temperature
          (
            tol,
            stk::simd::get_data(uOppNodeTangential, si),
            tempPlanar,
            currSurfaceTemperature,
            stk::simd::get_data(zh, si),
            mo::psim_unstable<double>,
            mo::psih_unstable<double>,
            stk::simd::get_data(kappa, si),
            stk::simd::get_data(z0, si),
            stk::simd::get_data(gravity, si),
            stk::simd::get_data(Tref, si),
            stk::simd::get_data(beta_m, si),
            stk::simd::get_data(beta_h, si)
          );
          std::cout << "utau = " << mean_fluxes_given_surf_temp[0] << " qWall = " << mean_fluxes_given_surf_temp[1] << std::endl;
*/






          const double Tflux1 = stk::simd::get_data(Tflux, si);
          double utau;
          if (Tflux1 < -eps) {
            utau = calc_utau(
              stk::simd::get_data(uTangential, si),
              stk::simd::get_data(zh, si),
              stk::simd::get_data(term, si),
              stk::simd::get_data(Tflux, si),
              stk::simd::get_data(Lfac, si),
              stk::simd::get_data(kappa, si),
              mo::psim_stable<double>,
              stk::simd::get_data(beta_m, si));
          } else if (Tflux1 > eps) {
            utau = calc_utau(
              stk::simd::get_data(uTangential, si),
              stk::simd::get_data(zh, si),
              stk::simd::get_data(term, si),
              stk::simd::get_data(Tflux, si),
              stk::simd::get_data(Lfac, si),
              stk::simd::get_data(kappa, si),
              mo::psim_unstable<double>,
              stk::simd::get_data(gamma_m, si));
          } else {
            utau = stk::simd::get_data(kappa, si) *
                   stk::simd::get_data(uTangential, si) /
                   stk::simd::get_data(term, si);
          }

          stk::simd::set_data(utau_calc, si, utau);
        }
        utauOps(feData, ip) = utau_calc;

        // Accumulate utau for statistics output
        uSum.array_[0] += utau_calc * aMag;
        uSum.array_[1] += aMag;
      }
    }, utauReducer);

  algDriver_.accumulate_utau_area_sum(utauSum.array_[0], utauSum.array_[1]);
}

INSTANTIATE_KERNEL_FACE_ELEMENT(ABLWallFluxesAlg)

}  // nalu
}  // sierra
