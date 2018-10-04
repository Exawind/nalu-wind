/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "wind_energy/BdyLayerStatistics.h"
#include "wind_energy/BdyHeightAlgorithm.h"
#include "Realm.h"
#include "TurbulenceAveragingPostProcessing.h"
#include "AveragingInfo.h"
#include "NaluEnv.h"
#include "utils/LinearInterpolation.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include "netcdf.h"

#include <cmath>
#include <fstream>
#include <string>

namespace sierra {
namespace nalu {

inline void check_nc_error(int code, std::string msg)
{
  if (code != 0)
    throw std::runtime_error("BdyLayerStatistics:: NetCDF error: " + msg);
}

BdyLayerStatistics::BdyLayerStatistics(
  Realm& realm,
  const YAML::Node& node
) : realm_(realm),
    nDim_(realm.spatialDimension_)
{
  load(node);
}

BdyLayerStatistics::~BdyLayerStatistics()
{}

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
    throw std::runtime_error("BdyLayerStatistics::load(): Incorrect height algorithm.");
  }

  double timeAvgWindow = 3600.0;
  get_if_present(node, "time_filter_interval", timeAvgWindow, timeAvgWindow);
  get_if_present(node, "compute_temperature_statistics", calcTemperatureStats_,
                 calcTemperatureStats_);

  setup_turbulence_averaging(timeAvgWindow);

  get_if_present(node, "output_frequency", outputFrequency_, outputFrequency_);
  get_if_present(node, "time_hist_output_frequency",
                 timeHistOutFrequency_, timeHistOutFrequency_);
  get_if_present(node, "stats_output_file", bdyStatsFile_, bdyStatsFile_);
  get_if_present(node, "process_utau_statistics", hasUTau_, hasUTau_);
}

void
BdyLayerStatistics::setup_turbulence_averaging(
  const double timeAvgWindow)
{
  bool hasTurbAvg = false;
  if (realm_.turbulenceAveragingPostProcessing_ == nullptr) {
    realm_.turbulenceAveragingPostProcessing_ = new TurbulenceAveragingPostProcessing(realm_);
  } else {
    hasTurbAvg = true;
  }

  auto* turbAvg = realm_.turbulenceAveragingPostProcessing_;

  if (hasTurbAvg) {
    const double diff = std::fabs(timeAvgWindow - turbAvg->timeFilterInterval_);
    if (diff > 1.0e-3)
      NaluEnv::self().naluOutputP0()
        << "WARNING:: BdyLayerStatistics: timeFilterInterval inconsistent with that requested for TurbulenceAveragingPostProcessing." << std::endl;
  } else {
    turbAvg->timeFilterInterval_ = timeAvgWindow;
    turbAvg->averagingType_ = TurbulenceAveragingPostProcessing::MOVING_EXPONENTIAL;
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

  for (size_t i=0; i < nparts; i++) {
    auto* part = meta.get_part(realm_.physics_part_name(partNames_[i]));
    if (nullptr == part)
      throw std::runtime_error("BdyLayerStatistics:: Part not found: " + partNames_[i]);
    else
      fluidParts_[i] = part;
  }

  heightIndex_ = &meta.declare_field<ScalarIntFieldType>(
    stk::topology::NODE_RANK, "bdy_layer_height_index_field");
  for (auto* part: fluidParts_)
    stk::mesh::put_field_on_mesh(*heightIndex_, *part, nullptr);
}

void
BdyLayerStatistics::initialize()
{
  auto& meta = realm_.meta_data();
  stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(fluidParts_);

  bdyHeightAlg_->calc_height_levels(sel, *heightIndex_, heights_);

  const size_t nHeights = heights_.size();
  sumVol_.resize(nHeights);
  rhoAvg_.resize(nHeights);
  velAvg_.resize(nHeights * nDim_);
  velBarAvg_.resize(nHeights * nDim_);
  uiujAvg_.resize(nHeights * nDim_ * 2);
  uiujBarAvg_.resize(nHeights * nDim_ * 2);
  sfsBarAvg_.resize(nHeights * nDim_ * 2);


  if (calcTemperatureStats_) {
    thetaAvg_.resize(nHeights);
    thetaBarAvg_.resize(nHeights);
    thetaUjAvg_.resize(nHeights * nDim_);
    thetaSFSBarAvg_.resize(nHeights * nDim_);
    thetaUjBarAvg_.resize(nHeights * nDim_);
    thetaVarAvg_.resize(nHeights);
    thetaBarVarAvg_.resize(nHeights);
  }

  // Time history output in a NetCDF file
  prepare_nc_file();

  doInit_ = false;
}

void
BdyLayerStatistics::execute()
{
  if (doInit_) initialize();

  compute_velocity_stats();
  output_velocity_averages();

  if (calcTemperatureStats_) {
    compute_temperature_stats();
    output_temperature_averages();
  }

  write_time_hist_file();
}

void
BdyLayerStatistics::velocity(
  double height,
  double* velVector)
{
  interpolate_variable(
    realm_.meta_data().spatial_dimension(),
    velAvg_, height, velVector);
}

void
BdyLayerStatistics::time_averaged_velocity(
  double height,
  double* velVector)
{
  interpolate_variable(
    realm_.meta_data().spatial_dimension(),
    velBarAvg_, height, velVector);
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

void
BdyLayerStatistics::interpolate_variable(
  int nComp,
  std::vector<double>& varArray,
  double height,
  double* interpVar)
{
  auto idx = utils::find_index(heights_, height);

  switch (idx.first) {
  case utils::OutOfBounds::LOWLIM: {
    int offset = idx.second * nComp;
    for (int d=0; d < nComp; d++) {
      interpVar[d] = varArray[offset + d];
    }
    break;
  }

  case utils::OutOfBounds::UPLIM: {
    int offset = (idx.second - 1) * nComp;
    for (int d=0; d < nComp; d++) {
      interpVar[d] = varArray[offset + d];
    }
    break;
  }

  case utils::OutOfBounds::VALID: {
    int ih = idx.second;
    int offset = idx.second * nComp;
    double fac = (height - heights_[ih]) / (heights_[ih+1] - heights_[ih]);
    for (int d=0; d < nComp; d++) {
      interpVar[d] = (1.0 - fac) * varArray[offset+d] + fac * varArray[offset + nComp + d];
    }
    break;
  }
  }
}

void
BdyLayerStatistics::compute_velocity_stats()
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(fluidParts_)
    & !(realm_.get_inactive_selector());

  const auto bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);

  ScalarFieldType* density = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  VectorFieldType* velocity = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity");
  VectorFieldType* velTimeAvg = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity_resa_abl");
  stk::mesh::FieldBase* resStress = meta.get_field(
    stk::topology::NODE_RANK, "resolved_stress");
  stk::mesh::FieldBase* sfsField = meta.get_field(
    stk::topology::NODE_RANK, "sfs_stress");
  stk::mesh::FieldBase* dualVol = meta.get_field(
    stk::topology::NODE_RANK, "dual_nodal_volume");

  const size_t nHeights = heights_.size();

  // Reset rows before accumulation
  for (size_t ih=0; ih < nHeights; ih++) {
    int offset = ih * nDim_;

    for (int d=0; d < nDim_; d++) {
      velAvg_[offset + d] = 0.0;
      velBarAvg_[offset + d] = 0.0;
    }

    offset *= 2;
    for (int i=0; i < nDim_ * 2; i++) {
      sfsBarAvg_[offset + i] = 0.0;
      uiujBarAvg_[offset + i] = 0.0;
      uiujAvg_[offset + i] = 0.0;
    }

    // Store sum volumes for temperature stats (processed next)
    sumVol_[ih] = 0.0;
    rhoAvg_[ih] = 0.0;
  }

  // Sum up all the local contributions
  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      int ih = *stk::mesh::field_data(*heightIndex_, node);
      int offset = ih * nDim_;

      // Volume calcs
      double dVol = *(double*)stk::mesh::field_data(*dualVol, node);
      sumVol_[ih] += dVol;

      // Density calcs
      double rho = *stk::mesh::field_data(*density, node);
      rhoAvg_[ih] += rho * dVol;

      // Velocity calculations
      double* vel = stk::mesh::field_data(*velocity, node);
      {
        double* velA = stk::mesh::field_data(*velTimeAvg, node);

        for (int d=0; d < nDim_; d++) {
          velAvg_[offset + d] += vel[d] * rho * dVol;
          velBarAvg_[offset + d] += velA[d] * dVol;
        }
      }

      // Stress calculations
      offset *= 2;
      {
        int idx = 0;
        for (int i=0; i < nDim_; i++) {
          for (int j=i; j < nDim_; j++) {
            uiujAvg_[offset + idx] += vel[i] * vel[j] * rho * dVol;
            idx++;
          }
        }

        double* sfs = static_cast<double*>(stk::mesh::field_data(*sfsField, node));
        double* uiuj = static_cast<double*>(stk::mesh::field_data(*resStress, node));

        for (int i=0; i < nDim_ * 2; i++) {
            sfsBarAvg_[offset + i] += sfs[i] * dVol;
            uiujBarAvg_[offset + i] += uiuj[i] * dVol;
        }
      }
    }
  }

  // Global summation
  MPI_Allreduce(MPI_IN_PLACE, velAvg_.data(), nHeights * nDim_, MPI_DOUBLE,
                MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, velBarAvg_.data(), nHeights * nDim_,
                MPI_DOUBLE, MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, sfsBarAvg_.data(), nHeights * nDim_ * 2,
                MPI_DOUBLE, MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, uiujBarAvg_.data(), nHeights * nDim_ * 2,
                MPI_DOUBLE, MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, uiujAvg_.data(), nHeights * nDim_ * 2,
                MPI_DOUBLE, MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, sumVol_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
                bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, rhoAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
                bulk.parallel());

  // Compute averages
  for (size_t ih=0; ih < nHeights; ih++) {
    int offset = ih * nDim_;

    for (int d=0; d < nDim_; d++) {
      velAvg_[offset + d] /= rhoAvg_[ih];
      velBarAvg_[offset + d] /= rhoAvg_[ih];
    }

    offset *= 2;
    for (int i=0; i < nDim_ * 2; i++) {
      sfsBarAvg_[offset + i] /= rhoAvg_[ih];
      uiujBarAvg_[offset + i] /= rhoAvg_[ih];
      uiujAvg_[offset + i] /= rhoAvg_[ih];
    }

    // Store density for temperature stats (processed next)
    rhoAvg_[ih] /= sumVol_[ih];
  }

  // Compute prime quantities
  for (size_t ih=0; ih < nHeights; ih++) {
    int offset = ih * nDim_;
    int offset1 = offset * 2;
    int idx = 0;

    for (int i=0; i < nDim_; i++) {
      for (int j=i; j < nDim_; j++) {
        uiujAvg_[offset1 + idx] -= velAvg_[offset + i] * velAvg_[offset + j];
        uiujBarAvg_[offset1 + idx] -= velBarAvg_[offset + i] * velBarAvg_[offset + j];
        idx++;
      }
    }
  }
}

void
BdyLayerStatistics::compute_temperature_stats()
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(fluidParts_)
    & !(realm_.get_inactive_selector());

  const auto bkts = bulk.get_buckets(stk::topology::NODE_RANK, sel);
  const size_t nHeights = heights_.size();

  ScalarFieldType* density = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  VectorFieldType* velocity = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity");
  ScalarFieldType* theta = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "temperature");
  ScalarFieldType* thetaA = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "temperature_resa_abl");
  ScalarFieldType* dualVol = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "dual_nodal_volume");
  stk::mesh::FieldBase* thetaSFS = meta.get_field(
    stk::topology::NODE_RANK, "temperature_sfs_flux");
  stk::mesh::FieldBase* thetaUj = meta.get_field(
    stk::topology::NODE_RANK, "temperature_resolved_flux");
  stk::mesh::FieldBase* thetaVar = meta.get_field(
    stk::topology::NODE_RANK, "temperature_variance");

  // Reset arrays before accumulation
  for (size_t ih=0; ih < nHeights; ih++) {
    thetaAvg_[ih] = 0.0;
    thetaBarAvg_[ih] = 0.0;
    thetaVarAvg_[ih] = 0.0;
    thetaBarVarAvg_[ih] = 0.0;

    int offset = ih * nDim_;
    for (int d=0; d < nDim_; d++) {
      thetaSFSBarAvg_[offset + d] = 0.0;
      thetaUjBarAvg_[offset + d] = 0.0;
      thetaUjAvg_[offset + d] = 0.0;
    }
  }

  // Sum up all local contributions
  for (auto b: bkts) {
    for (size_t in=0; in < b->size(); in++) {
      auto node = (*b)[in];
      int ih = *stk::mesh::field_data(*heightIndex_, node);

      // Temperature calculations
      double* rho = stk::mesh::field_data(*density, node);
      double* temp = stk::mesh::field_data(*theta, node);
      double* tempA = stk::mesh::field_data(*thetaA, node);
      double* dVol = stk::mesh::field_data(*dualVol, node);
      double* tVar = static_cast<double*>(stk::mesh::field_data(*thetaVar, node));

      thetaAvg_[ih] += rho[0] * temp[0] * dVol[0];
      thetaBarAvg_[ih] += tempA[0] * dVol[0];
      thetaVarAvg_[ih] += rho[0] * temp[0] * temp[0] * dVol[0];
      thetaBarVarAvg_[ih] += tVar[0] * dVol[0];

      double* vel = static_cast<double*>(stk::mesh::field_data(*velocity, node));
      double* tsfs = static_cast<double*>(stk::mesh::field_data(*thetaSFS, node));
      double* tuj = static_cast<double*>(stk::mesh::field_data(*thetaUj, node));

      auto offset = ih * nDim_;
      for (int d=0; d < nDim_; d++) {
        thetaSFSBarAvg_[offset + d] += tsfs[d] * dVol[0];
        thetaUjBarAvg_[offset + d] += tuj[d] * dVol[0];

        thetaUjAvg_[offset + d] += rho[0] * temp[0] * vel[d] * dVol[0];
      }
    }
  }

  // Global summation
  MPI_Allreduce(MPI_IN_PLACE, thetaAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
                bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, thetaBarAvg_.data(), nHeights, MPI_DOUBLE, MPI_SUM,
                bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, thetaSFSBarAvg_.data(), nHeights * nDim_,
                MPI_DOUBLE, MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, thetaUjAvg_.data(), nHeights * nDim_, MPI_DOUBLE,
                MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, thetaUjBarAvg_.data(), nHeights * nDim_,
                MPI_DOUBLE, MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, thetaVarAvg_.data(), nHeights, MPI_DOUBLE,
                MPI_SUM, bulk.parallel());
  MPI_Allreduce(MPI_IN_PLACE, thetaBarVarAvg_.data(), nHeights, MPI_DOUBLE,
                MPI_SUM, bulk.parallel());

  // Compute averages
  for (size_t ih=0; ih < nHeights; ih++) {
    double denom = (rhoAvg_[ih] * sumVol_[ih]);
    thetaAvg_[ih] /= denom;
    thetaBarAvg_[ih] /= denom;
    thetaVarAvg_[ih] /= denom;
    thetaBarVarAvg_[ih] /= denom;

    int offset = ih * nDim_;
    for (int d=0; d < nDim_; d++) {
      thetaSFSBarAvg_[offset + d] /= denom;
      thetaUjBarAvg_[offset + d] /= denom;
      thetaUjAvg_[offset + d] /= denom;
    }
  }

  for (size_t ih=0; ih < nHeights; ih++) {
    int offset = ih * nDim_;
    thetaVarAvg_[ih] -= thetaAvg_[ih] * thetaAvg_[ih];
    thetaBarVarAvg_[ih] -= thetaBarAvg_[ih] * thetaBarAvg_[ih];
    for (int d=0; d < nDim_; d++) {
      thetaUjAvg_[offset + d] -= thetaAvg_[ih] * velAvg_[offset + d];
      thetaUjBarAvg_[offset + d] -= thetaBarAvg_[ih] * velBarAvg_[offset + d];
    }
  }
}

void
BdyLayerStatistics::output_velocity_averages()
{
  const int tStep = realm_.get_time_step_count();
  const int iproc = realm_.bulk_data().parallel_rank();

  // Only output data if at the desired timestep
  if ((iproc != 0) || (tStep % outputFrequency_ != 0)) return;

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
  for (size_t ih=0; ih < nHeights; ih++) {
    int offset = ih * nDim_;

    // Velocity output
    velfile << heights_[ih];
    for (int d=0; d < nDim_; d++)
      velfile << " " << velBarAvg_[offset + d];
    for (int d=0; d < nDim_; d++)
      velfile << " " << velAvg_[offset + d];
    velfile << " " << rhoAvg_[ih] << std::endl;

    // Resolved and SFS stress outputs
    offset *= 2;
    sfsfile << heights_[ih];
    uiujfile << heights_[ih];
    for (int i=0; i < nDim_ * 2; i++) {
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
  if ((iproc != 0) || (tStep % outputFrequency_ != 0)) return;

  std::ofstream tempfile;
  tempfile.open("abl_temperature_stats.dat", std::ofstream::out);

  std::string curTime = std::to_string(realm_.get_current_time());
  tempfile << "# Time = " << curTime << std::endl;
  tempfile << "# Height, <T>, T, T' sqr" << std::endl;

  const size_t nHeights = heights_.size();
  for (size_t ih=0; ih < nHeights; ih++) {
    // temperature outputs
    tempfile << heights_[ih] << " "
             << thetaBarAvg_[ih] << " "
             << thetaAvg_[ih] << " "
             << thetaVarAvg_[ih] << std::endl;
  }

  tempfile.close();
}

void
BdyLayerStatistics::prepare_nc_file()
{
  const int iproc = realm_.bulk_data().parallel_rank();
  // Only process stats in the master MPI rank
  if (iproc != 0) return;

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
  ierr = nc_def_var(ncid, "velocity_tavg", NC_DOUBLE, 3, vecDims.data(), &varid);
  ncVarIDs_["velocity_tavg"] = varid;
  ierr = nc_def_var(ncid, "sfs_stress", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["sfs_stress"] = varid;
  ierr = nc_def_var(ncid, "resolved_stress", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["resolved_stress"] = varid;
  ierr = nc_def_var(ncid, "sfs_stress_tavg", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["sfs_stress_tavg"] = varid;
  ierr = nc_def_var(ncid, "resolved_stress_tavg", NC_DOUBLE, 3, stDims.data(), &varid);
  ncVarIDs_["resolved_stress_tavg"] = varid;

  if (calcTemperatureStats_) {
    ierr = nc_def_var(ncid, "temperature", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature"] = varid;
    ierr = nc_def_var(ncid, "temperature_tavg", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature_tavg"] = varid;
    ierr = nc_def_var(ncid, "temperature_sfs_flux_tavg", NC_DOUBLE, 3, vecDims.data(), &varid);
    ncVarIDs_["temperature_sfs_flux_tavg"] = varid;
    ierr = nc_def_var(ncid, "temperature_resolved_flux", NC_DOUBLE, 3, vecDims.data(), &varid);
    ncVarIDs_["temperature_resolved_flux"] = varid;
    ierr = nc_def_var(ncid, "temperature_variance", NC_DOUBLE, 2, twoDims.data(), &varid);
    ncVarIDs_["temperature_variance"] = varid;
    ierr = nc_def_var(ncid, "temperature_resolved_flux_tavg", NC_DOUBLE, 3, vecDims.data(), &varid);
    ncVarIDs_["temperature_resolved_flux_tavg"] = varid;
    ierr = nc_def_var(ncid, "temperature_variance_tavg", NC_DOUBLE, 2, twoDims.data(), &varid);
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
  if ((iproc != 0) || (tStep % timeHistOutFrequency_ != 0)) return;

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

  ierr = nc_put_vara_double(ncid, ncVarIDs_["time"], &tCount, &count0, &curTime);
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
      ncid, ncVarIDs_["temperature_sfs_flux_tavg"], start2.data(), count2.data(),
      thetaSFSBarAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_resolved_flux_tavg"], start2.data(),
      count2.data(), thetaUjBarAvg_.data());
    ierr = nc_put_vara_double(
      ncid, ncVarIDs_["temperature_variance_tavg"], start1.data(), count1.data(),
      thetaBarVarAvg_.data());
  }

  if (hasUTau_) {
    ierr = nc_put_vara_double(ncid, ncVarIDs_["utau"], &tCount, &count0, &uTauAvg_);
  }

  ierr = nc_close(ncid);
}

}  // nalu
}  // sierra
