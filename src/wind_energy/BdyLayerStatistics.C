// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "wind_energy/BdyLayerStatistics.h"
#include "wind_energy/BdyHeightAlgorithm.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "NaluParsing.h"
#include "Realm.h"
#include "TurbulenceAveragingPostProcessing.h"
#include "AveragingInfo.h"
#include "NaluEnv.h"
#include "utils/LinearInterpolation.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_mesh/base/NgpMesh.hpp"

#include "netcdf.h"

#include <cmath>
#include <fstream>
#include <string>

namespace sierra {
namespace nalu {

namespace {

inline typename utils::InterpTraits<double>::index_type
check_bounds(const BdyLayerStatistics::HostArrayType& xinp, const double& x)
{
  auto sz = xinp.size();

  if (sz < 2) {
    throw std::runtime_error(
      "Interpolation table contains less than 2 entries.");
  }

  if (x < xinp[0]) {
    return std::make_pair(utils::OutOfBounds::LOWLIM, 0);
  } else if (x > xinp[sz - 1]) {
    return std::make_pair(utils::OutOfBounds::UPLIM, sz - 1);
  } else {
    return std::make_pair(utils::OutOfBounds::VALID, 0);
  }
}

inline utils::InterpTraits<double>::index_type
find_index(const BdyLayerStatistics::HostArrayType& xinp, const double& x)
{
  auto idx = check_bounds(xinp, x);
  if (
    idx.first == utils::OutOfBounds::UPLIM ||
    idx.first == utils::OutOfBounds::LOWLIM)
    return idx;

  auto sz = xinp.size();
  for (size_t i = 1; i < sz; i++) {
    if (x <= xinp[i]) {
      idx.second = i - 1;
      break;
    }
  }
  return idx;
}

} // namespace

inline void
check_nc_error(int code, std::string msg)
{
  if (code != 0)
    throw std::runtime_error("BdyLayerStatistics:: NetCDF error: " + msg);
}

BdyLayerStatistics::BdyLayerStatistics(Realm& realm, const YAML::Node& node)
  : realm_(realm), nDim_(realm.spatialDimension_)
{
  load(node);
}

BdyLayerStatistics::~BdyLayerStatistics() {}

void
BdyLayerStatistics::load(const YAML::Node& node)
{
  const auto& partNames = node["target_name"];
  if (partNames.Type() == YAML::NodeType::Scalar) {
    auto pName = partNames.as<std::string>();
    partNames_.push_back(pName);
  } else {
    partNames_ = partNames.as<std::vector<std::string>>();
  }

  // Process algorithm used to determine unique heights
  std::string heightAlg = "rectilinear_mesh";
  get_if_present(node, "height_calc_algorithm", heightAlg, heightAlg);

  if (heightAlg == "rectilinear_mesh") {
    bdyHeightAlg_.reset(new RectilinearMeshHeightAlg(realm_, node));
  } else {
    throw std::runtime_error(
      "BdyLayerStatistics::load(): Incorrect height algorithm.");
  }

  double timeAvgWindow = 3600.0;
  get_if_present(node, "time_filter_interval", timeAvgWindow, timeAvgWindow);
  get_if_present(
    node, "compute_temperature_statistics", calcTemperatureStats_,
    calcTemperatureStats_);

  setup_turbulence_averaging(timeAvgWindow);

  get_if_present(node, "output_frequency", outputFrequency_, outputFrequency_);
  get_if_present(
    node, "time_hist_output_frequency", timeHistOutFrequency_,
    timeHistOutFrequency_);
  get_if_present(node, "stats_output_file", bdyStatsFile_, bdyStatsFile_);
  get_if_present(node, "process_utau_statistics", hasUTau_, hasUTau_);
}

void
BdyLayerStatistics::setup_turbulence_averaging(const double timeAvgWindow)
{
  bool hasTurbAvg = false;
  if (realm_.turbulenceAveragingPostProcessing_ == nullptr) {
    realm_.turbulenceAveragingPostProcessing_ =
      new TurbulenceAveragingPostProcessing(realm_);
  } else {
    hasTurbAvg = true;
  }

  auto* turbAvg = realm_.turbulenceAveragingPostProcessing_;

  if (hasTurbAvg) {
    const double diff = std::fabs(timeAvgWindow - turbAvg->timeFilterInterval_);
    if (diff > 1.0e-3)
      NaluEnv::self().naluOutputP0()
        << "WARNING:: BdyLayerStatistics: timeFilterInterval inconsistent with "
           "that requested for TurbulenceAveragingPostProcessing."
        << std::endl;
  } else {
    turbAvg->timeFilterInterval_ = timeAvgWindow;
    turbAvg->averagingType_ =
      TurbulenceAveragingPostProcessing::MOVING_EXPONENTIAL;
  }

  AveragingInfo* avInfo = new AveragingInfo();

  avInfo->name_ = "abl";
  avInfo->targetNames_ = partNames_;
  avInfo->computeSFSStress_ = true;
  avInfo->computeResolvedStress_ = true;
  avInfo->resolvedFieldNameVec_.push_back("velocity");

  if (calcTemperatureStats_) {
    avInfo->computeTemperatureResolved_ = true;
    avInfo->computeTemperatureSFS_ = true;
    avInfo->resolvedFieldNameVec_.push_back("temperature");
  }

  turbAvg->averageInfoVec_.push_back(avInfo);
}

void
BdyLayerStatistics::setup()
{
  auto& meta = realm_.meta_data();
  const size_t nparts = partNames_.size();
  fluidParts_.resize(nparts);

  for (size_t i = 0; i < nparts; i++) {
    auto* part = meta.get_part(realm_.physics_part_name(partNames_[i]));
    if (nullptr == part)
      throw std::runtime_error(
        "BdyLayerStatistics:: Part not found: " + partNames_[i]);
    else
      fluidParts_[i] = part;
  }

  heightIndex_ = &meta.declare_field<int>(
    stk::topology::NODE_RANK, "bdy_layer_height_index_field");
  for (auto* part : fluidParts_)
    stk::mesh::put_field_on_mesh(*heightIndex_, *part, nullptr);
}

void
BdyLayerStatistics::initialize()
{
  auto& meta = realm_.meta_data();
  stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(fluidParts_);

  std::vector<double> heights_vec;
  bdyHeightAlg_->calc_height_levels(sel, *heightIndex_, heights_vec);

  const size_t nHeights = heights_vec.size();
  d_heights_ = ArrayType("d_heights_", nHeights);
  d_sumVol_ = ArrayType("d_sumVol_", nHeights);
  d_rhoAvg_ = ArrayType("d_rhoAvg_", nHeights);
  d_velAvg_ = ArrayType("d_velAvg_", nHeights * nDim_);
  d_velMagAvg_ = ArrayType("d_velMagAvg_", nHeights);
  d_velBarAvg_ = ArrayType("d_velBarAvg_", nHeights * nDim_);
  d_uiujAvg_ = ArrayType("d_uiujAvg_", nHeights * nDim_ * 2);
  d_uiujBarAvg_ = ArrayType("d_uiujBarAvg_", nHeights * nDim_ * 2);
  d_sfsBarAvg_ = ArrayType("d_sfsBarAvg_", nHeights * nDim_ * 2);
  d_sfsAvg_ = ArrayType("d_sfsAvg_", nHeights * nDim_ * 2);

  heights_ = Kokkos::create_mirror_view(d_heights_);
  sumVol_ = Kokkos::create_mirror_view(d_sumVol_);
  rhoAvg_ = Kokkos::create_mirror_view(d_rhoAvg_);
  velAvg_ = Kokkos::create_mirror_view(d_velAvg_);
  velMagAvg_ = Kokkos::create_mirror_view(d_velMagAvg_);
  velBarAvg_ = Kokkos::create_mirror_view(d_velBarAvg_);
  uiujAvg_ = Kokkos::create_mirror_view(d_uiujAvg_);
  uiujBarAvg_ = Kokkos::create_mirror_view(d_uiujBarAvg_);
  sfsBarAvg_ = Kokkos::create_mirror_view(d_sfsBarAvg_);
  sfsAvg_ = Kokkos::create_mirror_view(d_sfsAvg_);

  if (calcTemperatureStats_) {
    d_thetaAvg_ = ArrayType("thetaAvg_", nHeights);
    d_thetaBarAvg_ = ArrayType("thetaBarAvg_", nHeights);
    d_thetaUjAvg_ = ArrayType("thetaUjAvg_", nHeights * nDim_);
    d_thetaSFSBarAvg_ = ArrayType("thetaSFSBarAvg_", nHeights * nDim_);
    d_thetaUjBarAvg_ = ArrayType("thetaUjBarAvg_", nHeights * nDim_);
    d_thetaVarAvg_ = ArrayType("thetaVarAvg_", nHeights);
    d_thetaBarVarAvg_ = ArrayType("thetaBarVarAvg_", nHeights);

    thetaAvg_ = Kokkos::create_mirror_view(d_thetaAvg_);
    thetaBarAvg_ = Kokkos::create_mirror_view(d_thetaBarAvg_);
    thetaUjAvg_ = Kokkos::create_mirror_view(d_thetaUjAvg_);
    thetaSFSBarAvg_ = Kokkos::create_mirror_view(d_thetaSFSBarAvg_);
    thetaUjBarAvg_ = Kokkos::create_mirror_view(d_thetaUjBarAvg_);
    thetaVarAvg_ = Kokkos::create_mirror_view(d_thetaVarAvg_);
    thetaBarVarAvg_ = Kokkos::create_mirror_view(d_thetaBarVarAvg_);
  }

  // Copy heights into the Kokkos views
  for (size_t ih = 0; ih < nHeights; ++ih)
    heights_[ih] = heights_vec[ih];
  Kokkos::deep_copy(d_heights_, heights_);

  // Time history output in a NetCDF file
  prepare_nc_file();

  doInit_ = false;
}

void
BdyLayerStatistics::execute()
{
  if (doInit_)
    initialize();

  impl_compute_velocity_stats();
  output_velocity_averages();

  if (calcTemperatureStats_) {
    impl_compute_temperature_stats();
    output_temperature_averages();
  }

  write_time_hist_file();
}

void
BdyLayerStatistics::velocity(double height, double* velVector)
{
  interpolate_variable(
    realm_.meta_data().spatial_dimension(), velAvg_, height, velVector);
}

void
BdyLayerStatistics::time_averaged_velocity(double height, double* velVector)
{
  interpolate_variable(
    realm_.meta_data().spatial_dimension(), velBarAvg_, height, velVector);
}

void
BdyLayerStatistics::velocity_magnitude(double height, double* velMag)
{
  interpolate_variable(1, velMagAvg_, height, velMag);
}

void
BdyLayerStatistics::density(double height, double* rho)
{
  interpolate_variable(1, rhoAvg_, height, rho);
}

void
BdyLayerStatistics::temperature(double height, double* theta)
{
  interpolate_variable(1, thetaAvg_, height, theta);
}

int
BdyLayerStatistics::abl_height_index(const double height) const
{
  auto idx = find_index(heights_, height);

  return idx.second;
}

void
BdyLayerStatistics::interpolate_variable(
  int nComp, HostArrayType& varArray, double height, double* interpVar)
{
  auto idx = find_index(heights_, height);

  switch (idx.first) {
  case utils::OutOfBounds::LOWLIM: {
    int offset = idx.second * nComp;
    for (int d = 0; d < nComp; d++) {
      interpVar[d] = varArray[offset + d];
    }
    break;
  }

  case utils::OutOfBounds::UPLIM: {
    int offset = (idx.second - 1) * nComp;
    for (int d = 0; d < nComp; d++) {
      interpVar[d] = varArray[offset + d];
    }
    break;
  }

  case utils::OutOfBounds::VALID: {
    int ih = idx.second;
    int offset = idx.second * nComp;
    double fac = (height - heights_[ih]) / (heights_[ih + 1] - heights_[ih]);
    for (int d = 0; d < nComp; d++) {
      interpVar[d] =
        (1.0 - fac) * varArray[offset + d] + fac * varArray[offset + nComp + d];
    }
    break;
  }
  }
}

void
BdyLayerStatistics::impl_compute_velocity_stats()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();
  const auto density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto velTimeAvg =
    nalu_ngp::get_ngp_field(meshInfo, "velocity_resa_abl");
  const auto resStress = nalu_ngp::get_ngp_field(meshInfo, "resolved_stress");
  const auto sfsField = nalu_ngp::get_ngp_field(meshInfo, "sfs_stress");
  const auto sfsFieldInst =
    nalu_ngp::get_ngp_field(meshInfo, "sfs_stress_inst");
  const auto dualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  const auto heightIndex = realm_.ngp_field_manager().get_field<int>(
    heightIndex_->mesh_meta_data_ordinal());

  stk::mesh::Selector sel = realm_.meta_data().locally_owned_part() &
                            stk::mesh::selectUnion(fluidParts_) &
                            !(realm_.get_inactive_selector()) &
                            !(realm_.replicated_periodic_node_selector());

  // Reset arrays before accumulation
  Kokkos::deep_copy(d_velAvg_, 0.0);
  Kokkos::deep_copy(d_velMagAvg_, 0.0);
  Kokkos::deep_copy(d_velBarAvg_, 0.0);
  Kokkos::deep_copy(d_sfsBarAvg_, 0.0);
  Kokkos::deep_copy(d_sfsAvg_, 0.0);
  Kokkos::deep_copy(d_uiujAvg_, 0.0);
  Kokkos::deep_copy(d_uiujBarAvg_, 0.0);
  Kokkos::deep_copy(d_sumVol_, 0.0);
  Kokkos::deep_copy(d_rhoAvg_, 0.0);

  // Bring arrays into local scope for capture on device
  auto d_velAvg = d_velAvg_;
  auto d_velMagAvg = d_velMagAvg_;
  auto d_velBarAvg = d_velBarAvg_;
  auto d_sfsBarAvg = d_sfsBarAvg_;
  auto d_uiujAvg = d_uiujAvg_;
  auto d_uiujBarAvg = d_uiujBarAvg_;
  auto d_sumVol = d_sumVol_;
  auto d_rhoAvg = d_rhoAvg_;
  auto d_sfsAvg = d_sfsAvg_;

  const int ndim = nDim_;
  nalu_ngp::run_entity_algorithm(
    "BLStats::velocity", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const int ih = heightIndex.get(mi, 0);

      // Volume and density calculations
      const double rho = density.get(mi, 0);
      const double dVol = dualVol.get(mi, 0);
      Kokkos::atomic_add(&d_sumVol(ih), (dVol));
      Kokkos::atomic_add(&d_rhoAvg(ih), (rho * dVol));

      // Velocity computations
      int offset = ih * ndim;

      // -this is the horizontal velocity magnitude--needs to be generalized to
      // let the user specify if it
      //  should just be horizontal, what the horizontal plane is, or the full
      //  vector magnitude.  This implementation assumes horizontal is in
      //  Cartesian x and y.
      double velMag = 0.0;
      for (int d = 0; d < ndim - 1; ++d) {
        velMag += velocity.get(mi, d) * velocity.get(mi, d);
      }
      velMag = stk::math::sqrt(velMag);
      Kokkos::atomic_add(&d_velMagAvg(ih), (velMag * rho * dVol));

      for (int d = 0; d < ndim; ++d) {
        Kokkos::atomic_add(
          &d_velAvg(offset + d), (velocity.get(mi, d) * rho * dVol));

        // velocity_resa_abl is already multiplied by density
        Kokkos::atomic_add(
          &d_velBarAvg(offset + d), (velTimeAvg.get(mi, d) * dVol));
      }

      // Stress computations
      offset *= 2;
      int idx = 0;
      for (int i = 0; i < ndim; ++i)
        for (int j = i; j < ndim; ++j) {
          Kokkos::atomic_add(
            &d_uiujAvg(offset + idx),
            (velocity.get(mi, i) * velocity.get(mi, j) * rho * dVol));
          idx++;
        }

      for (int i = 0; i < ndim * 2; ++i) {
        Kokkos::atomic_add(
          &d_sfsAvg(offset + i), (sfsFieldInst.get(mi, i) * rho * dVol));
        Kokkos::atomic_add(
          &d_sfsBarAvg(offset + i), (sfsField.get(mi, i) * dVol));
        Kokkos::atomic_add(
          &d_uiujBarAvg(offset + i), (resStress.get(mi, i) * dVol));
      }
    });

  Kokkos::deep_copy(velAvg_, d_velAvg_);
  Kokkos::deep_copy(velMagAvg_, d_velMagAvg_);
  Kokkos::deep_copy(velBarAvg_, d_velBarAvg_);
  Kokkos::deep_copy(sfsBarAvg_, d_sfsBarAvg_);
  Kokkos::deep_copy(sfsAvg_, d_sfsAvg_);
  Kokkos::deep_copy(uiujAvg_, d_uiujAvg_);
  Kokkos::deep_copy(uiujBarAvg_, d_uiujBarAvg_);
  Kokkos::deep_copy(sumVol_, d_sumVol_);
  Kokkos::deep_copy(rhoAvg_, d_rhoAvg_);

  // Global summation
  const size_t nHeights = heights_.extent(0);
  const auto& bulk = realm_.bulk_data();
  MPI_Allreduce(
    MPI_IN_PLACE, velAvg_.data(), nHeights * nDim_, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, velMagAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, velBarAvg_.data(), nHeights * nDim_, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, sfsBarAvg_.data(), nHeights * nDim_ * 2, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, sfsAvg_.data(), nHeights * nDim_ * 2, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, uiujBarAvg_.data(), nHeights * nDim_ * 2, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, uiujAvg_.data(), nHeights * nDim_ * 2, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, sumVol_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, rhoAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());

  // Compute averages
  for (size_t ih = 0; ih < nHeights; ih++) {
    int offset = ih * nDim_;

    for (int d = 0; d < nDim_; d++) {
      velAvg_(offset + d) /= rhoAvg_(ih);
      velBarAvg_(offset + d) /= rhoAvg_(ih);
    }

    velMagAvg_(ih) /= rhoAvg_(ih);

    offset *= 2;
    for (int i = 0; i < nDim_ * 2; i++) {
      sfsBarAvg_(offset + i) /= rhoAvg_(ih);
      sfsAvg_(offset + i) /= rhoAvg_(ih);
      uiujBarAvg_(offset + i) /= rhoAvg_(ih);
      uiujAvg_(offset + i) /= rhoAvg_(ih);
    }

    // Store density for temperature stats (processed next)
    rhoAvg_(ih) /= sumVol_(ih);
  }

  // Compute prime quantities
  for (size_t ih = 0; ih < nHeights; ih++) {
    int offset = ih * nDim_;
    int offset1 = offset * 2;
    int idx = 0;

    for (int i = 0; i < nDim_; i++) {
      for (int j = i; j < nDim_; j++) {
        uiujAvg_(offset1 + idx) -= velAvg_(offset + i) * velAvg_(offset + j);
        uiujBarAvg_(offset1 + idx) -=
          velBarAvg_(offset + i) * velBarAvg_(offset + j);
        idx++;
      }
    }
  }
}

void
BdyLayerStatistics::impl_compute_temperature_stats()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex;
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = realm_.ngp_mesh();

  const auto density = nalu_ngp::get_ngp_field(meshInfo, "density");
  const auto velocity = nalu_ngp::get_ngp_field(meshInfo, "velocity");
  const auto theta = nalu_ngp::get_ngp_field(meshInfo, "temperature");
  const auto thetaA = nalu_ngp::get_ngp_field(meshInfo, "temperature_resa_abl");
  const auto dualVol = nalu_ngp::get_ngp_field(meshInfo, "dual_nodal_volume");
  const auto thetaSFS =
    nalu_ngp::get_ngp_field(meshInfo, "temperature_sfs_flux");
  const auto thetaUj =
    nalu_ngp::get_ngp_field(meshInfo, "temperature_resolved_flux");
  const auto thetaVar =
    nalu_ngp::get_ngp_field(meshInfo, "temperature_variance");
  const auto heightIndex = realm_.ngp_field_manager().get_field<int>(
    heightIndex_->mesh_meta_data_ordinal());

  stk::mesh::Selector sel = realm_.meta_data().locally_owned_part() &
                            stk::mesh::selectUnion(fluidParts_) &
                            !(realm_.get_inactive_selector()) &
                            !(realm_.replicated_periodic_node_selector());

  // Reset arrays before accumulation
  Kokkos::deep_copy(d_thetaAvg_, 0.0);
  Kokkos::deep_copy(d_thetaBarAvg_, 0.0);
  Kokkos::deep_copy(d_thetaVarAvg_, 0.0);
  Kokkos::deep_copy(d_thetaBarVarAvg_, 0.0);
  Kokkos::deep_copy(d_thetaSFSBarAvg_, 0.0);
  Kokkos::deep_copy(d_thetaUjBarAvg_, 0.0);
  Kokkos::deep_copy(d_thetaUjAvg_, 0.0);

  // Bring arrays into local scope for capture on device
  ArrayType d_thetaAvg = d_thetaAvg_;
  ArrayType d_thetaBarAvg = d_thetaBarAvg_;
  ArrayType d_thetaVarAvg = d_thetaVarAvg_;
  ArrayType d_thetaBarVarAvg = d_thetaBarVarAvg_;
  ArrayType d_thetaSFSBarAvg = d_thetaSFSBarAvg_;
  ArrayType d_thetaUjBarAvg = d_thetaUjBarAvg_;
  ArrayType d_thetaUjAvg = d_thetaUjAvg_;

  const int ndim = nDim_;
  nalu_ngp::run_entity_algorithm(
    "BLStats::temperature", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      const int ih = heightIndex.get(mi, 0);

      const double rho = density.get(mi, 0);
      const double dVol = dualVol.get(mi, 0);

      Kokkos::atomic_add(&d_thetaAvg(ih), (rho * theta.get(mi, 0) * dVol));
      Kokkos::atomic_add(&d_thetaBarAvg(ih), (thetaA.get(mi, 0) * dVol));
      Kokkos::atomic_add(
        &d_thetaVarAvg(ih), (rho * theta.get(mi, 0) * theta.get(mi, 0) * dVol));
      Kokkos::atomic_add(&d_thetaBarVarAvg(ih), (thetaVar.get(mi, 0) * dVol));

      const int offset = ih * ndim;
      for (int d = 0; d < ndim; ++d) {
        Kokkos::atomic_add(
          &d_thetaSFSBarAvg(offset + d), (thetaSFS.get(mi, d) * dVol));
        Kokkos::atomic_add(
          &d_thetaUjBarAvg(offset + d), (thetaUj.get(mi, d) * dVol));
        Kokkos::atomic_add(
          &d_thetaUjAvg(offset + d),
          (rho * theta.get(mi, 0) * velocity.get(mi, d) * dVol));
      }
    });

  // Copy back to host
  Kokkos::deep_copy(thetaAvg_, d_thetaAvg_);
  Kokkos::deep_copy(thetaBarAvg_, d_thetaBarAvg_);
  Kokkos::deep_copy(thetaVarAvg_, d_thetaVarAvg_);
  Kokkos::deep_copy(thetaBarVarAvg_, d_thetaBarVarAvg_);
  Kokkos::deep_copy(thetaSFSBarAvg_, d_thetaSFSBarAvg_);
  Kokkos::deep_copy(thetaUjBarAvg_, d_thetaUjBarAvg_);
  Kokkos::deep_copy(thetaUjAvg_, d_thetaUjAvg_);

  // Global summation
  const size_t nHeights = heights_.extent(0);
  const auto& bulk = realm_.bulk_data();
  MPI_Allreduce(
    MPI_IN_PLACE, thetaAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, thetaBarAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, thetaSFSBarAvg_.data(), nHeights * nDim_, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, thetaUjAvg_.data(), nHeights * nDim_, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, thetaUjBarAvg_.data(), nHeights * nDim_, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, thetaVarAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());
  MPI_Allreduce(
    MPI_IN_PLACE, thetaBarVarAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
    bulk.parallel());

  // Compute averages
  for (size_t ih = 0; ih < nHeights; ih++) {
    double denom = (rhoAvg_(ih) * sumVol_(ih));
    thetaAvg_(ih) /= denom;
    thetaBarAvg_(ih) /= denom;
    thetaVarAvg_(ih) /= denom;
    thetaBarVarAvg_(ih) /= denom;

    int offset = ih * nDim_;
    for (int d = 0; d < nDim_; d++) {
      thetaSFSBarAvg_(offset + d) /= denom;
      thetaUjBarAvg_(offset + d) /= denom;
      thetaUjAvg_(offset + d) /= denom;
    }
  }

  // Compute primes
  for (size_t ih = 0; ih < nHeights; ih++) {
    int offset = ih * nDim_;
    thetaVarAvg_(ih) -= thetaAvg_(ih) * thetaAvg_(ih);
    thetaBarVarAvg_(ih) -= thetaBarAvg_(ih) * thetaBarAvg_(ih);
    for (int d = 0; d < nDim_; d++) {
      thetaUjAvg_(offset + d) -= thetaAvg_(ih) * velAvg_(offset + d);
      thetaUjBarAvg_(offset + d) -= thetaBarAvg_(ih) * velBarAvg_(offset + d);
    }
  }
}

void
BdyLayerStatistics::output_velocity_averages()
{
  const int tStep = realm_.get_time_step_count();
  const int iproc = realm_.bulk_data().parallel_rank();

  // Only output data if at the desired timestep
  if ((iproc != 0) || (tStep % outputFrequency_ != 0))
    return;

  std::ofstream velfile;
  std::ofstream uiujfile;
  std::ofstream sfsfile;

  // TODO: Allow customizable filenames?
  velfile.open("abl_velocity_stats.dat", std::ofstream::out);
  uiujfile.open("abl_resolved_stress_stats.dat", std::ofstream::out);
  sfsfile.open("abl_sfs_stress_stats.dat", std::ofstream::out);

  std::string curTime = std::to_string(realm_.get_current_time());
  velfile << "# Time = " << curTime << std::endl;
  uiujfile << "# Time = " << curTime << std::endl;
  sfsfile << "# Time = " << curTime << std::endl;
  velfile << "# Height, <Ux>, <Uy>, <Uz>, Ux, Uy, Uz, rho" << std::endl;
  uiujfile << "# Height, u11, u12, u13, u22, u23, u33" << std::endl;
  sfsfile << "# Height, t11, t12, t13, t22, t23, t33" << std::endl;

  // TODO: Fix format/precision options for output
  const size_t nHeights = heights_.size();
  for (size_t ih = 0; ih < nHeights; ih++) {
    int offset = ih * nDim_;

    // Velocity output
    velfile << heights_[ih];
    for (int d = 0; d < nDim_; d++)
      velfile << " " << velBarAvg_[offset + d];
    for (int d = 0; d < nDim_; d++)
      velfile << " " << velAvg_[offset + d];
    velfile << " " << rhoAvg_[ih] << std::endl;

    // Resolved and SFS stress outputs
    offset *= 2;
    sfsfile << heights_[ih];
    uiujfile << heights_[ih];
    for (int i = 0; i < nDim_ * 2; i++) {
      sfsfile << " " << sfsBarAvg_[offset + i];
      uiujfile << " " << uiujAvg_[offset + i];
    }
    sfsfile << std::endl;
    uiujfile << std::endl;
  }

  velfile.close();
  sfsfile.close();
  uiujfile.close();
}

void
BdyLayerStatistics::output_temperature_averages()
{
  const int tStep = realm_.get_time_step_count();
  const int iproc = realm_.bulk_data().parallel_rank();

  // Only output data if at the desired timestep
  if ((iproc != 0) || (tStep % outputFrequency_ != 0))
    return;

  std::ofstream tempfile;
  tempfile.open("abl_temperature_stats.dat", std::ofstream::out);

  std::string curTime = std::to_string(realm_.get_current_time());
  tempfile << "# Time = " << curTime << std::endl;
  tempfile << "# Height, <T>, T, T' sqr" << std::endl;

  const size_t nHeights = heights_.size();
  for (size_t ih = 0; ih < nHeights; ih++) {
    // temperature outputs
    tempfile << heights_[ih] << " " << thetaBarAvg_[ih] << " " << thetaAvg_[ih]
             << " " << thetaVarAvg_[ih] << std::endl;
  }

  tempfile.close();
}

void
BdyLayerStatistics::prepare_nc_file()
{
  const int iproc = realm_.bulk_data().parallel_rank();
  // Only process stats in the master MPI rank
  if (iproc != 0)
    return;

  int ncid, recDim, htDim, vDim, stDim, varid;
  int ierr;

  const int nHeights = heights_.size();

  // Create the file
  ierr = nc_create(bdyStatsFile_.c_str(), NC_CLOBBER, &ncid);
  check_nc_error(ierr, "nc_create");

  // Define dimensions for the NetCDF file
  ierr = nc_def_dim(ncid, "num_timesteps", NC_UNLIMITED, &recDim);
  ierr = nc_def_dim(ncid, "num_heights", nHeights, &htDim);
  ierr = nc_def_dim(ncid, "vec_dim", nDim_, &vDim);
  ierr = nc_def_dim(ncid, "stress_dim", nDim_ * 2, &stDim);

  // Define variables, we will store NetCDF IDs for the variables in the
  // ncVarIDs_ mapping for later use
  const std::vector<int> twoDims{recDim, htDim};
  const std::vector<int> vecDims{recDim, htDim, vDim};
  const std::vector<int> stDims{recDim, htDim, stDim};

  ierr = nc_def_var(ncid, "time", NC_DOUBLE, 1, &recDim, &varid);
  ncVarIDs_["time"] = varid;
  ierr = nc_def_var(ncid, "heights", NC_DOUBLE, 1, &htDim, &varid);
  ncVarIDs_["heights"] = varid;

  ierr = nc_def_var(ncid, "density", NC_DOUBLE, 2, twoDims.data(), &varid);
  ncVarIDs_["density"] = varid;
  ierr = nc_def_var(ncid, "velocity", NC_DOUBLE, 3, vecDims.data(), &varid);
  ncVarIDs_["velocity"] = varid;
  ierr =
    nc_def_var(ncid, "velocity_tavg", NC_DOUBLE, 3, vecDims.data(), &varid);
  ncVarIDs_["velocity_tavg"] = varid;
  ierr = nc_def_var(ncid, "sfs_stress", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["sfs_stress"] = varid;
  ierr =
    nc_def_var(ncid, "resolved_stress", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["resolved_stress"] = varid;
  ierr =
    nc_def_var(ncid, "sfs_stress_tavg", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["sfs_stress_tavg"] = varid;
  ierr = nc_def_var(
    ncid, "resolved_stress_tavg", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["resolved_stress_tavg"] = varid;

  if (calcTemperatureStats_) {
    ierr =
      nc_def_var(ncid, "temperature", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature"] = varid;
    ierr = nc_def_var(
      ncid, "temperature_tavg", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature_tavg"] = varid;
    ierr = nc_def_var(
      ncid, "temperature_sfs_flux_tavg", NC_DOUBLE, 3, vecDims.data(), &varid);
    ncVarIDs_["temperature_sfs_flux_tavg"] = varid;
    ierr = nc_def_var(
      ncid, "temperature_resolved_flux", NC_DOUBLE, 3, vecDims.data(), &varid);
    ncVarIDs_["temperature_resolved_flux"] = varid;
    ierr = nc_def_var(
      ncid, "temperature_variance", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature_variance"] = varid;
    ierr = nc_def_var(
      ncid, "temperature_resolved_flux_tavg", NC_DOUBLE, 3, vecDims.data(),
      &varid);
    ncVarIDs_["temperature_resolved_flux_tavg"] = varid;
    ierr = nc_def_var(
      ncid, "temperature_variance_tavg", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature_variance_tavg"] = varid;
  }

  if (hasUTau_) {
    ierr = nc_def_var(ncid, "utau", NC_DOUBLE, 1, &recDim, &varid);
    ncVarIDs_["utau"] = varid;
  }

  //! Indicate that we are done defining variables, ready to write data
  ierr = nc_enddef(ncid);
  check_nc_error(ierr, "nc_enddef");

  //! Populate height array upon initialization
  ierr = nc_put_var_double(ncid, ncVarIDs_["heights"], heights_.data());
  ierr = nc_close(ncid);
  check_nc_error(ierr, "nc_close");

  // Initialze the starting timestep
  startStep_ = realm_.get_time_step_count();
}

void
BdyLayerStatistics::write_time_hist_file()
{
  const int tStep = realm_.get_time_step_count() - startStep_;
  const int iproc = realm_.bulk_data().parallel_rank();

  // Only output data if at the desired timestep
  if ((iproc != 0) || (tStep % timeHistOutFrequency_ != 0))
    return;

  int ncid, ierr;
  const size_t nHeights = heights_.size();
  const size_t nDim = static_cast<size_t>(nDim_);
  const size_t tCount = tStep / timeHistOutFrequency_;
  const double curTime = realm_.get_current_time();

  ierr = nc_open(bdyStatsFile_.c_str(), NC_WRITE, &ncid);
  check_nc_error(ierr, "nc_open");
  ierr = nc_enddef(ncid);

  size_t count0 = 1;
  std::vector<size_t> start1{tCount, 0};
  std::vector<size_t> count1{1, nHeights};
  std::vector<size_t> start2{tCount, 0, 0};
  std::vector<size_t> count2{1, nHeights, nDim};
  std::vector<size_t> start3{tCount, 0, 0};
  std::vector<size_t> count3{1, nHeights, nDim * 2};

  ierr =
    nc_put_vara_double(ncid, ncVarIDs_["time"], &tCount, &count0, &curTime);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["density"], start1.data(), count1.data(), rhoAvg_.data());
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["velocity"], start2.data(), count2.data(), velAvg_.data());
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["resolved_stress"], start3.data(), count3.data(),
    uiujAvg_.data());
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["velocity_tavg"], start2.data(), count2.data(),
    velBarAvg_.data());
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["sfs_stress_tavg"], start3.data(), count3.data(),
    sfsBarAvg_.data());
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["sfs_stress"], start3.data(), count3.data(),
    sfsAvg_.data());
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["resolved_stress_tavg"], start3.data(), count3.data(),
    uiujBarAvg_.data());

  if (calcTemperatureStats_) {
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature"], start1.data(), count1.data(),
      thetaAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_resolved_flux"], start2.data(),
      count2.data(), thetaUjAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_variance"], start1.data(), count1.data(),
      thetaVarAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_tavg"], start1.data(), count1.data(),
      thetaBarAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_sfs_flux_tavg"], start2.data(),
      count2.data(), thetaSFSBarAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_resolved_flux_tavg"], start2.data(),
      count2.data(), thetaUjBarAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_variance_tavg"], start1.data(),
      count1.data(), thetaBarVarAvg_.data());
  }

  if (hasUTau_) {
    ierr =
      nc_put_vara_double(ncid, ncVarIDs_["utau"], &tCount, &count0, &uTauAvg_);
  }

  ierr = nc_close(ncid);
}

} // namespace nalu
} // namespace sierra
