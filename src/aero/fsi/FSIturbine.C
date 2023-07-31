// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "aero/fsi/FSIturbine.h"
#include "aero/aero_utils/DeflectionRamping.h"
#include "aero/aero_utils/ForceMoment.h"
#include "aero/fsi/MapLoad.h"
#include "aero/aero_utils/Pt2Line.h"
#include "utils/ComputeVectorDivergence.h"
#include <NaluEnv.h>
#include <NaluParsing.h>

#include "stk_util/parallel/ParallelReduce.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_math/StkMath.hpp"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"

#include "netcdf.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace sierra {

namespace nalu {

inline void
check_nc_error(int code, std::string msg)
{
  if (code != 0)
    throw std::runtime_error("BdyLayerStatistics:: NetCDF error: " + msg);
}

fsiTurbine::fsiTurbine(int iTurb, const YAML::Node& node)
  : iTurb_(iTurb),
    turbineProc_(-1),
    turbineInProc_(false),
    loadMap_(NULL),
    dispMap_(NULL),
    deflectionRamp_(NULL),
    dispMapInterp_(NULL),
    tforceSCS_(NULL)
{

  if (node["tower_parts"]) {
    const auto& tparts = node["tower_parts"];
    twrPartNames_ = tparts.as<std::vector<std::string>>();
  } else
    NaluEnv::self().naluOutputP0()
      << "Tower part name(s) not specified for turbine " << iTurb_ << std::endl;

  if (node["nacelle_parts"]) {
    const auto& nparts = node["nacelle_parts"];
    nacellePartNames_ = nparts.as<std::vector<std::string>>();
  } else
    NaluEnv::self().naluOutputP0()
      << "Nacelle part name(s) not specified for turbine " << iTurb_
      << std::endl;

  if (node["hub_parts"]) {
    const auto& hparts = node["hub_parts"];
    hubPartNames_ = hparts.as<std::vector<std::string>>();
  } else
    NaluEnv::self().naluOutputP0()
      << "Hub part name(s) not specified for turbine " << iTurb_ << std::endl;

  if (node["blade_parts"]) {
    const auto& bparts = node["blade_parts"];
    nBlades_ = bparts.size();
    bladePartNames_.resize(nBlades_);
    bladeParts_.resize(nBlades_);
    for (int iBlade = 0; iBlade < nBlades_; iBlade++) {
      const auto& bpart = bparts[iBlade];
      bladePartNames_[iBlade] = bpart.as<std::vector<std::string>>();
    }
    // --------------------------------------------------------------------------
    // Displacement Ramping
    // --------------------------------------------------------------------------
    const YAML::Node defNode = node["deflection_ramping"];
    ThrowErrorMsgIf(
      !defNode,
      "defleciton_ramping inputs are required for FSI Turbines with blades");
    DeflectionRampingParams& defParams = deflectionRampParams_;
    double* zeroTheta = &defParams.zeroRampLocTheta_;
    double* thetaRamp = &defParams.thetaRampSpan_;
    // clang-format off
    get_required(defNode, "temporal_ramp_start", defParams.startTimeTemporalRamp_);
    get_required(defNode, "temporal_ramp_end",   defParams.endTimeTemporalRamp_);
    get_required(defNode, "span_ramp_distance",  defParams.spanRampDistance_);
    get_if_present(defNode, "zero_theta_ramp_angle", *zeroTheta, *zeroTheta);
    get_if_present(defNode, "theta_ramp_span",       *thetaRamp, *thetaRamp);
    // clang-format on
    // ---------- conversionions ----------
    defParams.zeroRampLocTheta_ = utils::radians(defParams.zeroRampLocTheta_);
    defParams.thetaRampSpan_ = utils::radians(defParams.thetaRampSpan_);
    // --------------------------------------------------------------------------
  } else
    NaluEnv::self().naluOutputP0()
      << "Blade part names not specified for turbine " << iTurb_ << std::endl;

  if (node["tower_boundary_parts"]) {
    const auto& tparts = node["tower_boundary_parts"];
    twrBndyPartNames_ = tparts.as<std::vector<std::string>>();
    bndryPartNames_.insert(
      bndryPartNames_.begin(), twrBndyPartNames_.begin(),
      twrBndyPartNames_.end());
  } else
    NaluEnv::self().naluOutputP0()
      << "Tower boundary part name(s) not specified for turbine " << iTurb_
      << std::endl;

  if (node["nacelle_boundary_parts"]) {
    const auto& nparts = node["nacelle_boundary_parts"];
    nacelleBndyPartNames_ = nparts.as<std::vector<std::string>>();
    bndryPartNames_.insert(
      bndryPartNames_.end(), nacelleBndyPartNames_.begin(),
      nacelleBndyPartNames_.end());
  } else
    NaluEnv::self().naluOutputP0()
      << "Nacelle boundary part name(s) not specified for turbine " << iTurb_
      << std::endl;

  if (node["hub_boundary_parts"]) {
    const auto& hparts = node["hub_boundary_parts"];
    hubBndyPartNames_ = hparts.as<std::vector<std::string>>();
    bndryPartNames_.insert(
      bndryPartNames_.begin(), hubBndyPartNames_.begin(),
      hubBndyPartNames_.end());
  } else
    NaluEnv::self().naluOutputP0()
      << "Hub boundary part name(s) not specified for turbine " << iTurb_
      << std::endl;

  if (node["blade_boundary_parts"]) {
    const auto& bparts = node["blade_boundary_parts"];
    nBlades_ = bparts.size();
    bladeBndyPartNames_.resize(nBlades_);
    bladeBndyParts_.resize(nBlades_);
    for (int iBlade = 0; iBlade < nBlades_; iBlade++) {
      const auto& bpart = bparts[iBlade];
      bladeBndyPartNames_[iBlade] = bpart.as<std::vector<std::string>>();
      bndryPartNames_.insert(
        bndryPartNames_.begin(), bladeBndyPartNames_[iBlade].begin(),
        bladeBndyPartNames_[iBlade].end());
    }
  } else
    NaluEnv::self().naluOutputP0()
      << "Blade boundary part names not specified for turbine " << iTurb_
      << std::endl;
}

fsiTurbine::~fsiTurbine()
{

  // Nothing to do here so far
}

void
fsiTurbine::populateParts(
  std::vector<std::string>& partNames,
  stk::mesh::PartVector& partVec,
  stk::mesh::PartVector& allPartVec,
  const std::string& turbinePart)
{

  auto& meta = bulk_->mesh_meta_data();

  for (auto pName : partNames) {
    stk::mesh::Part* part = meta.get_part(pName);
    if (nullptr == part) {
      throw std::runtime_error(
        "fsiTurbine:: No part found for " + turbinePart +
        " mesh part corresponding to " + pName);
    } else {
      partVec.push_back(part);
      allPartVec.push_back(part);
    }

    stk::mesh::put_field_on_mesh(*dispMap_, *part, 1, nullptr);
    stk::mesh::put_field_on_mesh(*dispMapInterp_, *part, 1, nullptr);
    stk::mesh::put_field_on_mesh(*deflectionRamp_, *part, 1, nullptr);
  }
}

void
fsiTurbine::populateBndyParts(
  std::vector<std::string>& partNames,
  stk::mesh::PartVector& partVec,
  stk::mesh::PartVector& allPartVec,
  const std::string& turbinePart)
{

  auto& meta = bulk_->mesh_meta_data();

  for (auto pName : partNames) {
    stk::mesh::Part* part = meta.get_part(pName);
    if (nullptr == part) {
      throw std::runtime_error(
        "fsiTurbine:: No part found for " + turbinePart +
        " mesh part corresponding to " + pName);
    } else {
      partVec.push_back(part);
      allPartVec.push_back(part);
    }

    // TODO: Get number of SCS's per face from stk::topology and MasterElement.
    //  Currently assumes all-quad faces with 4 SCS's per face
    stk::mesh::put_field_on_mesh(*loadMap_, *part, 4, nullptr);
    stk::mesh::put_field_on_mesh(*loadMapInterp_, *part, 4, nullptr);
    stk::mesh::put_field_on_mesh(*tforceSCS_, *part, 4 * 3, nullptr);
  }
}

void
fsiTurbine::setup(std::shared_ptr<stk::mesh::BulkData> bulk)
{

  // TODO: Check if any part of the turbine surface is on this processor and set
  // turbineInProc_ to True/False

  // TODO:: Figure out a way to check the consistency between the number of
  // blades specified in the Nalu input file and the number of blades in the
  // OpenFAST model.

  bulk_ = bulk;
  auto& meta = bulk_->mesh_meta_data();

  deflectionRamp_ = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "deflection_ramp");
  if (deflectionRamp_ == NULL)
    deflectionRamp_ = &(meta.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "deflection_ramp"));

  dispMap_ =
    meta.get_field<ScalarIntFieldType>(stk::topology::NODE_RANK, "disp_map");
  if (dispMap_ == NULL)
    dispMap_ = &(meta.declare_field<ScalarIntFieldType>(
      stk::topology::NODE_RANK, "disp_map"));

  dispMapInterp_ = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "disp_map_interp");
  if (dispMapInterp_ == NULL)
    dispMapInterp_ = &(meta.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "disp_map_interp"));

  loadMap_ = meta.get_field<GenericIntFieldType>(meta.side_rank(), "load_map");
  if (loadMap_ == NULL)
    loadMap_ =
      &(meta.declare_field<GenericIntFieldType>(meta.side_rank(), "load_map"));

  loadMapInterp_ =
    meta.get_field<GenericFieldType>(meta.side_rank(), "load_map_interp");
  if (loadMapInterp_ == NULL)
    loadMapInterp_ = &(meta.declare_field<GenericFieldType>(
      meta.side_rank(), "load_map_interp"));

  tforceSCS_ = meta.get_field<GenericFieldType>(meta.side_rank(), "tforce_scs");
  if (tforceSCS_ == NULL)
    tforceSCS_ =
      &(meta.declare_field<GenericFieldType>(meta.side_rank(), "tforce_scs"));

  VectorFieldType* mesh_disp_ref = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement_ref");
  if (mesh_disp_ref == NULL)
    mesh_disp_ref = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_displacement_ref"));

  VectorFieldType* mesh_vel_ref = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity_ref");
  if (mesh_vel_ref == NULL)
    mesh_vel_ref = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "mesh_velocity_ref"));

  ScalarFieldType* div_mesh_vel = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_mesh_velocity");
  if (div_mesh_vel == NULL)
    div_mesh_vel = &(meta.declare_field<ScalarFieldType>(
      stk::topology::NODE_RANK, "div_mesh_velocity"));
  stk::mesh::put_field_on_mesh(
    *div_mesh_vel, meta.universal_part(), 1, nullptr);

  populateParts(twrPartNames_, twrParts_, partVec_, "Tower");
  populateParts(nacellePartNames_, nacelleParts_, partVec_, "Nacelle");
  populateParts(hubPartNames_, hubParts_, partVec_, "Hub");
  for (int i = 0; i < nBlades_; i++)
    populateParts(
      bladePartNames_[i], bladeParts_[i], partVec_,
      "Blade " + std::to_string(i));

  populateBndyParts(twrBndyPartNames_, twrBndyParts_, bndyPartVec_, "Tower");
  populateBndyParts(
    nacelleBndyPartNames_, nacelleBndyParts_, bndyPartVec_, "Nacelle");
  populateBndyParts(hubBndyPartNames_, hubBndyParts_, bndyPartVec_, "Hub");
  for (int i = 0; i < nBlades_; i++)
    populateBndyParts(
      bladeBndyPartNames_[i], bladeBndyParts_[i], bndyPartVec_,
      "Blade " + std::to_string(i));

  calc_loads_ = std::make_unique<CalcLoads>(bndyPartVec_);
  calc_loads_->setup(bulk_);
}

void
fsiTurbine::initialize()
{

  // Allocate memory for loads and deflections data

  int nTwrPts = params_.nBRfsiPtsTwr;
  int nBlades = params_.numBlades;
  int nTotBldPts = 0;
  for (int i = 0; i < nBlades; i++)
    nTotBldPts += params_.nBRfsiPtsBlade[i];
  brFSIdata_.twr_ref_pos.resize(6 * nTwrPts);
  brFSIdata_.twr_def.resize(6 * nTwrPts);
  brFSIdata_.twr_vel.resize(6 * nTwrPts);
  brFSIdata_.twr_ld.resize(6 * nTwrPts);
  brFSIdata_.bld_rloc.resize(nTotBldPts);
  brFSIdata_.bld_chord.resize(nTotBldPts);
  brFSIdata_.bld_ref_pos.resize(6 * nTotBldPts);
  brFSIdata_.bld_def.resize(6 * nTotBldPts);
  brFSIdata_.bld_vel.resize(6 * nTotBldPts);
  brFSIdata_.bld_ld.resize(6 * nTotBldPts);
  brFSIdata_.bld_root_ref_pos.resize(6 * nBlades);
  brFSIdata_.bld_root_def.resize(6 * nBlades);
  brFSIdata_.bld_pitch.resize(nBlades);
  brFSIdata_.hub_ref_pos.resize(6);
  brFSIdata_.hub_def.resize(6);
  brFSIdata_.hub_vel.resize(6);
  brFSIdata_.nac_ref_pos.resize(6);
  brFSIdata_.nac_def.resize(6);
  brFSIdata_.nac_vel.resize(6);

  bldDefStiff_.resize(nTotBldPts);
  bld_dr_.resize(nTotBldPts);
  bld_rmm_.resize(nTotBldPts);

  calc_loads_->initialize();
}

void
fsiTurbine::prepare_nc_file(
  const int nTwrPts, const int nBlades, const int nTotBldPts)
{

  const int iproc = bulk_->parallel_rank();
  if (iproc != turbineProc_)
    return;

  int ncid, n_dim, n_tsteps, n_twr_nds, n_blds, n_bld_nds, varid;
  int ierr;

  int nBldPts = nTotBldPts / nBlades;

  // Create the file
  std::stringstream defloads_fstream;
  defloads_fstream << "turb_";
  defloads_fstream << std::setfill('0') << std::setw(2) << iTurb_;
  defloads_fstream << "_deflloads.nc";
  std::string defloads_filename = defloads_fstream.str();
  ierr = nc_create(defloads_filename.c_str(), NC_CLOBBER, &ncid);
  check_nc_error(ierr, "nc_create");

  // Define dimensions
  ierr = nc_def_dim(ncid, "n_dim", 3, &n_dim);
  ierr = nc_def_dim(ncid, "n_tsteps", NC_UNLIMITED, &n_tsteps);
  ierr = nc_def_dim(ncid, "n_twr_nds", nTwrPts, &n_twr_nds);
  ierr = nc_def_dim(ncid, "n_blds", nBlades, &n_blds);
  ierr = nc_def_dim(ncid, "n_bld_nds", nBldPts, &n_bld_nds);

  const std::vector<int> twrRefDims{n_dim, n_twr_nds};
  const std::vector<int> twrDefLoadsDims{n_tsteps, n_dim, n_twr_nds};
  // const std::vector<int> bldRootRefDims{n_dim, n_blds};
  // const std::vector<int> bldRootDefDim{n_tsteps, n_dim, n_blds};
  const std::vector<int> bldParamDims{n_blds, n_bld_nds};
  const std::vector<int> bldRefDims{n_blds, n_dim, n_bld_nds};
  const std::vector<int> bldDefLoadsDims{n_tsteps, n_blds, n_dim, n_bld_nds};
  const std::vector<int> ptRefDims{n_dim};
  const std::vector<int> ptDefLoadsDims{n_tsteps, n_dim};

  // Now define variables
  ierr = nc_def_var(ncid, "time", NC_DOUBLE, 1, &n_tsteps, &varid);
  ncVarIDs_["time"] = varid;

  ierr =
    nc_def_var(ncid, "twr_ref_pos", NC_DOUBLE, 2, twrRefDims.data(), &varid);
  ncVarIDs_["twr_ref_pos"] = varid;
  ierr =
    nc_def_var(ncid, "twr_ref_orient", NC_DOUBLE, 2, twrRefDims.data(), &varid);
  ncVarIDs_["twr_ref_orient"] = varid;
  ierr =
    nc_def_var(ncid, "bld_chord", NC_DOUBLE, 2, bldParamDims.data(), &varid);
  ncVarIDs_["bld_chord"] = varid;
  ierr =
    nc_def_var(ncid, "bld_rloc", NC_DOUBLE, 2, bldParamDims.data(), &varid);
  ncVarIDs_["bld_rloc"] = varid;
  ierr =
    nc_def_var(ncid, "bld_ref_pos", NC_DOUBLE, 3, bldRefDims.data(), &varid);
  ncVarIDs_["bld_ref_pos"] = varid;
  ierr =
    nc_def_var(ncid, "bld_ref_orient", NC_DOUBLE, 3, bldRefDims.data(), &varid);
  ncVarIDs_["bld_ref_orient"] = varid;
  ierr =
    nc_def_var(ncid, "hub_ref_pos", NC_DOUBLE, 1, ptRefDims.data(), &varid);
  ncVarIDs_["hub_ref_pos"] = varid;
  ierr =
    nc_def_var(ncid, "hub_ref_orient", NC_DOUBLE, 1, ptRefDims.data(), &varid);
  ncVarIDs_["hub_ref_orient"] = varid;
  ierr =
    nc_def_var(ncid, "nac_ref_pos", NC_DOUBLE, 1, ptRefDims.data(), &varid);
  ncVarIDs_["nac_ref_pos"] = varid;
  ierr =
    nc_def_var(ncid, "nac_ref_orient", NC_DOUBLE, 1, ptRefDims.data(), &varid);
  ncVarIDs_["nac_ref_orient"] = varid;

  ierr =
    nc_def_var(ncid, "twr_disp", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
  ncVarIDs_["twr_disp"] = varid;
  ierr = nc_def_var(
    ncid, "twr_orient", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
  ncVarIDs_["twr_orient"] = varid;
  ierr =
    nc_def_var(ncid, "twr_vel", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
  ncVarIDs_["twr_vel"] = varid;
  ierr = nc_def_var(
    ncid, "twr_rotvel", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
  ncVarIDs_["twr_rotvel"] = varid;
  ierr =
    nc_def_var(ncid, "twr_ld", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
  ncVarIDs_["twr_ld"] = varid;
  ierr = nc_def_var(
    ncid, "twr_moment", NC_DOUBLE, 3, twrDefLoadsDims.data(), &varid);
  ncVarIDs_["twr_moment"] = varid;

  ierr =
    nc_def_var(ncid, "bld_disp", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_disp"] = varid;
  ierr = nc_def_var(
    ncid, "bld_orient", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_orient"] = varid;
  ierr =
    nc_def_var(ncid, "bld_vel", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_vel"] = varid;
  ierr = nc_def_var(
    ncid, "bld_rotvel", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_rotvel"] = varid;
  ierr =
    nc_def_var(ncid, "bld_ld", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_ld"] = varid;
  ierr = nc_def_var(
    ncid, "bld_ld_loc", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_ld_loc"] = varid;
  ierr = nc_def_var(
    ncid, "bld_moment", NC_DOUBLE, 4, bldDefLoadsDims.data(), &varid);
  ncVarIDs_["bld_moment"] = varid;

  ierr =
    nc_def_var(ncid, "hub_disp", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["hub_disp"] = varid;
  ierr =
    nc_def_var(ncid, "hub_orient", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["hub_orient"] = varid;
  ierr =
    nc_def_var(ncid, "hub_vel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["hub_vel"] = varid;
  ierr =
    nc_def_var(ncid, "hub_rotvel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["hub_rotvel"] = varid;

  ierr =
    nc_def_var(ncid, "nac_disp", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["nac_disp"] = varid;
  ierr =
    nc_def_var(ncid, "nac_orient", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["nac_orient"] = varid;
  ierr =
    nc_def_var(ncid, "nac_vel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["nac_vel"] = varid;
  ierr =
    nc_def_var(ncid, "nac_rotvel", NC_DOUBLE, 3, ptDefLoadsDims.data(), &varid);
  ncVarIDs_["nac_rotvel"] = varid;

  //! Indicate that we are done defining variables, ready to write data
  ierr = nc_enddef(ncid);
  check_nc_error(ierr, "nc_enddef");
  ierr = nc_close(ncid);
  check_nc_error(ierr, "nc_close");
}

void
fsiTurbine::write_nc_ref_pos()
{

  const int iproc = bulk_->parallel_rank();
  if (iproc != turbineProc_)
    return;

  size_t nTwrPts = params_.nBRfsiPtsTwr;
  size_t nBlades = params_.numBlades;
  size_t nTotBldPts = 0;
  for (size_t i = 0; i < nBlades; i++) {
    nTotBldPts += params_.nBRfsiPtsBlade[i];
  }
  size_t nBldPts = nTotBldPts / nBlades;
  int ncid, ierr;

  std::stringstream defloads_fstream;
  defloads_fstream << "turb_";
  defloads_fstream << std::setfill('0') << std::setw(2) << iTurb_;
  defloads_fstream << "_deflloads.nc";
  std::string defloads_filename = defloads_fstream.str();
  ierr = nc_open(defloads_filename.c_str(), NC_WRITE, &ncid);
  check_nc_error(ierr, "nc_open");
  ierr = nc_enddef(ncid);

  std::vector<double> tmpArray;

  tmpArray.resize(nTwrPts);
  {
    std::vector<size_t> count_dim{1, nTwrPts};
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_ref_pos[i * 6 + idim];
      std::vector<size_t> start_dim{idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_ref_pos"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_ref_pos[i * 6 + 3 + idim];
      std::vector<size_t> start_dim{idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_ref_orient"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
  }

  tmpArray.resize(nBldPts);
  {
    std::vector<size_t> count_dim{1, 1, nBldPts};
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_ref_pos[(iStart * 6) + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_ref_pos"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_ref_pos[(iStart * 6) + iDim + 3];
          iStart++;
        }
        std::vector<size_t> start_dim{iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_ref_orient"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }

    std::vector<size_t> param_count_dim{1, nBldPts};
    size_t iStart = 0;
    for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
      for (size_t i = 0; i < nBldPts; i++) {
        tmpArray[i] = brFSIdata_.bld_chord[iStart];
        iStart++;
      }
      std::vector<size_t> start_dim{iBlade, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["bld_chord"], start_dim.data(), param_count_dim.data(),
        tmpArray.data());
    }
    iStart = 0;
    for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
      for (size_t i = 0; i < nBldPts; i++) {
        tmpArray[i] = brFSIdata_.bld_rloc[iStart];
        iStart++;
      }
      std::vector<size_t> start_dim{iBlade, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["bld_rloc"], start_dim.data(), param_count_dim.data(),
        tmpArray.data());
    }
  }

  ierr = nc_put_var_double(
    ncid, ncVarIDs_["nac_ref_pos"], &brFSIdata_.nac_ref_pos[0]);
  ierr = nc_put_var_double(
    ncid, ncVarIDs_["nac_ref_orient"], &brFSIdata_.nac_ref_pos[3]);

  ierr = nc_put_var_double(
    ncid, ncVarIDs_["hub_ref_pos"], &brFSIdata_.hub_ref_pos[0]);
  ierr = nc_put_var_double(
    ncid, ncVarIDs_["hub_ref_orient"], &brFSIdata_.hub_ref_pos[3]);

  ierr = nc_close(ncid);
}

void
fsiTurbine::write_nc_def_loads(const size_t tStep, const double curTime)
{

  const int iproc = bulk_->parallel_rank();
  if (iproc != turbineProc_)
    return;

  size_t nTwrPts = params_.nBRfsiPtsTwr;
  size_t nBlades = params_.numBlades;
  size_t nTotBldPts = 0;
  for (size_t i = 0; i < nBlades; i++)
    nTotBldPts += params_.nBRfsiPtsBlade[i];
  size_t nBldPts = nTotBldPts / nBlades;

  int ncid, ierr;

  std::stringstream defloads_fstream;
  defloads_fstream << "turb_";
  defloads_fstream << std::setfill('0') << std::setw(2) << iTurb_;
  defloads_fstream << "_deflloads.nc";
  std::string defloads_filename = defloads_fstream.str();
  ierr = nc_open(defloads_filename.c_str(), NC_WRITE, &ncid);
  check_nc_error(ierr, "nc_open");
  ierr = nc_enddef(ncid);

  size_t iStart = 0;
  for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
    for (size_t i = 1; i < nBldPts - 1; i++) {
      brFSIdata_.bld_ld[(i + iStart) * 6 + 4] =
        (0.5 * (brFSIdata_.bld_rloc[i + iStart + 1] -
                brFSIdata_.bld_rloc[i + iStart - 1]));
    }
    brFSIdata_.bld_ld[(iStart)*6 + 4] =
      (0.5 * (brFSIdata_.bld_rloc[iStart + 1] - brFSIdata_.bld_rloc[iStart]));
    brFSIdata_.bld_ld[(iStart + nBldPts - 1) * 6 + 4] =
      (0.5 * (brFSIdata_.bld_rloc[iStart + nBldPts - 1] -
              brFSIdata_.bld_rloc[iStart + nBldPts - 2]));
    iStart += nBldPts;
  }

  std::ofstream csvOut;
  csvOut.open("bld_def.csv", std::ofstream::out);
  csvOut << "rloc, x, y, z, , chord" << std::endl;
  for (size_t i = 0; i < nTotBldPts; i++) {
    csvOut << brFSIdata_.bld_rloc[i] << ", "
           << brFSIdata_.bld_ref_pos[i * 6] + brFSIdata_.bld_def[i * 6] << ", "
           << brFSIdata_.bld_ref_pos[i * 6 + 1] + brFSIdata_.bld_def[i * 6 + 1]
           << ", "
           << brFSIdata_.bld_ref_pos[i * 6 + 2] + brFSIdata_.bld_def[i * 6 + 2]
           << ", " << brFSIdata_.bld_chord[i] << std::endl;
  }
  csvOut.close();

  size_t count0 = 1;
  const std::vector<size_t> twrDefLoadsDims{1, 6 * nTwrPts};
  // const std::vector<size_t> bldRootDefLoadsDims{1, 3*6*nBlades};
  const std::vector<size_t> bldDefLoadsDims{1, 6 * nTotBldPts};
  const std::vector<size_t> ptDefLoadsDims{1, 6};

  ierr = nc_put_vara_double(ncid, ncVarIDs_["time"], &tStep, &count0, &curTime);

  std::vector<double> tmpArray;

  tmpArray.resize(nTwrPts);
  {
    std::vector<size_t> count_dim{1, 1, nTwrPts};
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_def[i * 6 + idim];
      std::vector<size_t> start_dim{tStep, idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_disp"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_def[i * 6 + 3 + idim];
      std::vector<size_t> start_dim{tStep, idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_orient"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_vel[i * 6 + idim];
      std::vector<size_t> start_dim{tStep, idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_vel"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_def[i * 6 + 3 + idim];
      std::vector<size_t> start_dim{tStep, idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_rotvel"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }

    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_ld[i * 6 + idim];
      std::vector<size_t> start_dim{tStep, idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_ld"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
    for (size_t idim = 0; idim < 3; idim++) {
      for (size_t i = 0; i < nTwrPts; i++)
        tmpArray[i] = brFSIdata_.twr_ld[i * 6 + 3 + idim];
      std::vector<size_t> start_dim{tStep, idim, 0};
      ierr = nc_put_vara_double(
        ncid, ncVarIDs_["twr_moment"], start_dim.data(), count_dim.data(),
        tmpArray.data());
    }
  }

  tmpArray.resize(nBldPts);
  {
    std::vector<size_t> count_dim{1, 1, 1, nBldPts};
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_def[(iStart * 6) + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_disp"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_def[(iStart * 6) + 3 + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_orient"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_vel[(iStart * 6) + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_vel"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_vel[(iStart * 6) + 3 + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_rotvel"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_ld[(iStart * 6) + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_ld"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }

    std::vector<double> ld_loc(3 * nTotBldPts, 0.0);
    for (size_t i = 0; i < nTotBldPts; i++) {
      applyWMrotation(
        &brFSIdata_.bld_def[i * 6 + 3], &brFSIdata_.bld_ld[i * 6],
        &ld_loc[i * 3]);
    }
    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = ld_loc[iStart * 3 + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_ld_loc"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }

    for (size_t iDim = 0; iDim < 3; iDim++) {
      size_t iStart = 0;
      for (size_t iBlade = 0; iBlade < nBlades; iBlade++) {
        for (size_t i = 0; i < nBldPts; i++) {
          tmpArray[i] = brFSIdata_.bld_ld[(iStart * 6) + 3 + iDim];
          iStart++;
        }
        std::vector<size_t> start_dim{tStep, iBlade, iDim, 0};
        ierr = nc_put_vara_double(
          ncid, ncVarIDs_["bld_moment"], start_dim.data(), count_dim.data(),
          tmpArray.data());
      }
    }
  }

  std::vector<size_t> start_dim{tStep, 0};
  std::vector<size_t> count_dim{1, 3};
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["hub_disp"], start_dim.data(), count_dim.data(),
    &brFSIdata_.hub_def[0]);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["hub_orient"], start_dim.data(), count_dim.data(),
    &brFSIdata_.hub_def[3]);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["hub_vel"], start_dim.data(), count_dim.data(),
    &brFSIdata_.hub_vel[0]);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["hub_rotvel"], start_dim.data(), ptDefLoadsDims.data(),
    &brFSIdata_.hub_vel[3]);

  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["nac_disp"], start_dim.data(), count_dim.data(),
    &brFSIdata_.nac_def[0]);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["nac_orient"], start_dim.data(), count_dim.data(),
    &brFSIdata_.nac_def[3]);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["nac_vel"], start_dim.data(), count_dim.data(),
    &brFSIdata_.nac_vel[0]);
  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["nac_rotvel"], start_dim.data(), ptDefLoadsDims.data(),
    &brFSIdata_.nac_vel[3]);

  ierr = nc_close(ncid);
}

//! Convert pressure and viscous/turbulent stress on the turbine surface CFD
//! mesh into a "fsiForce" field on the turbine surface CFD mesh
void
fsiTurbine::computeFSIforce()
{
}

//! Map loads from the "fsiForce" field on the turbine surface CFD mesh into
//! point load array that gets transferred to openfast
void
fsiTurbine::mapLoads()
{

  calc_loads_->execute();

  // To implement this function - assume that 'loadMap_' field contains the node
  // id along the blade or the tower that will accumulate the load corresponding
  // to the node on the CFD surface mesh

  // First zero out forces on the OpenFAST mesh
  for (int i = 0; i < params_.nBRfsiPtsTwr; i++) {
    for (int j = 0; j < 6; j++)
      brFSIdata_.twr_ld[i * 6 + j] = 0.0;
  }

  int nBlades = params_.numBlades;
  int iRunTot = 0;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    for (int i = 0; i < nPtsBlade; i++) {
      for (int j = 0; j < 6; j++)
        brFSIdata_.bld_ld[iRunTot * 6 + j] = 0.0;
      iRunTot++;
    }
  }

  auto& meta = bulk_->mesh_meta_data();
  VectorFieldType* modelCoords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* meshDisp = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");

  fsi::mapTowerLoad(
    *bulk_, twrBndyParts_, *modelCoords, *meshDisp, *loadMap_, *loadMapInterp_,
    *tforceSCS_, brFSIdata_.twr_ref_pos, brFSIdata_.twr_def, brFSIdata_.twr_ld);

  // Now the blades
  int iStart = 0;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];

    fsi::mapBladeLoad(
      *bulk_, bladeBndyParts_[iBlade], *modelCoords, *meshDisp, *loadMap_,
      *loadMapInterp_, *tforceSCS_, nPtsBlade, iStart, brFSIdata_.bld_rloc,
      brFSIdata_.bld_ref_pos, brFSIdata_.bld_def, brFSIdata_.bld_ld);

    iStart += nPtsBlade;
  }
}

void
fsiTurbine::computeHubForceMomentForPart(
  std::vector<double>& hubForceMoment,
  std::vector<double>& hubPos,
  stk::mesh::PartVector partVec)
{

  auto& meta = bulk_->mesh_meta_data();
  VectorFieldType* modelCoords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* meshDisp = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");
  GenericFieldType* tforce =
    meta.get_field<GenericFieldType>(meta.side_rank(), "tforce_scs");
  std::vector<double> l_hubForceMoment(6, 0.0);
  std::array<double, 3> tmpMeshPos{
    0.0, 0.0, 0.0}; // Vector to temporarily store mesh node location

  // TODO: This is looping over the wrong buckets - Nodes instead of faces
  // Is this even required anymore? Probly can delete

  // stk::mesh::Selector sel(
  //   meta.locally_owned_part() & stk::mesh::selectUnion(partVec));
  // const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
  // for (auto b : bkts) {
  //   for (size_t in = 0; in < b->size(); in++) {
  //     auto node = (*b)[in];
  //     double* xyz = stk::mesh::field_data(*modelCoords, node);
  //     double* xyz_disp = stk::mesh::field_data(*meshDisp, node);
  //     double* pressureForceNode = stk::mesh::field_data(*pressureForce,
  //     node); double* viscForceNode = stk::mesh::field_data(*tauWall, node);
  //     std::array<double, 3> fsiForceNode;
  //     for (int i = 0; i < 3; i++) {
  //       fsiForceNode[i] = pressureForceNode[i] + viscForceNode[i];
  //       tmpMeshPos[i] = xyz[i] + xyz_disp[i];
  //     }
  //     fsi::computeEffForceMoment(
  //       fsiForceNode.data(), tmpMeshPos.data(), l_hubForceMoment.data(),
  //       hubPos.data());
  //   }
  // }

  stk::all_reduce_sum(
    bulk_->parallel(), l_hubForceMoment.data(), hubForceMoment.data(), 6);
}

//! Set displacement corresponding to rotation at a constant rpm on the OpenFAST
//! mesh before mapping to the turbine blade surface mesh
void
fsiTurbine::setRotationDisplacement(
  std::array<double, 3> axis, double omega, double curTime)
{

  double theta = omega * curTime;
  double twopi = 2.0 * M_PI;
  theta = std::fmod(theta, twopi);
  NaluEnv::self().naluOutputP0()
    << "Setting rotation of " << theta * 180.0 / M_PI << " degrees about ["
    << axis[0] << "," << axis[1] << "," << axis[2] << "]" << std::endl;

  // Rotate the hub first
  double hubRot = 4.0 * tan(0.25 * theta);
  std::vector<double> wmHubRot = {
    hubRot * axis[0], hubRot * axis[1], hubRot * axis[2]};
  for (size_t i = 0; i < 3; i++)
    brFSIdata_.hub_def[3 + i] = -wmHubRot[i];

  // For each node on the openfast blade1 mesh - compute distance from the blade
  // root node. Apply a rotation varying as the square of the distance between 0
  // - 45 degrees about the [0 1 0] axis. Apply a translation displacement that
  // produces a tip displacement of 5m
  int iStart = 0;
  int nBlades = params_.numBlades;
  ;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    double bladeRot = 4.0 * tan(0.25 * iBlade * 120.0 * M_PI / 180.0);
    std::vector<double> wmRotBlade_ref = {
      bladeRot * axis[0], bladeRot * axis[1], bladeRot * axis[2]};
    std::vector<double> wmRotBlade(3, 0.0);
    composeWM(wmHubRot.data(), wmRotBlade_ref.data(), wmRotBlade.data());

    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    for (int i = 0; i < nPtsBlade; i++) {

      // Transpose the whole thing
      for (int j = 0; j < 3; j++)
        brFSIdata_.bld_def[(iStart + i) * 6 + 3 + j] = -wmRotBlade[j];

      // Set translational displacement
      std::vector<double> r(3, 0.0);
      for (int j = 0; j < 3; j++)
        r[j] = brFSIdata_.bld_ref_pos[(iStart + i) * 6 + j] -
               brFSIdata_.hub_ref_pos[j];

      std::vector<double> rRot(3, 0.0);

      applyWMrotation(wmHubRot.data(), r.data(), rRot.data());
      brFSIdata_.bld_def[(iStart + i) * 6 + 0] = rRot[0] - r[0];
      brFSIdata_.bld_def[(iStart + i) * 6 + 1] = rRot[1] - r[1];
      brFSIdata_.bld_def[(iStart + i) * 6 + 2] = rRot[2] - r[2];
    }
    iStart += nPtsBlade;
  }
}

//! Set sample displacement on the OpenFAST mesh before mapping to the turbine
//! blade surface mesh
void
fsiTurbine::setSampleDisplacement(double curTime)
{

  /*
      Step 1: Get hub ref orientation - Apply to [1,0,0] to get hub rotation
     axis and calculate Wiener-Milenkovic parameter corresponding to hub
     rotation Step 2: Get Blade root ref position - Apply to [0,0,1] to get
     pitch axis. Calculate pitch deformation WM parameter. Set blade pitch Step
     3: For each blade node - calculate local deformation axis by applying WM
     corresponding to ref position to [0,0,1] Step 4: Create final deformation
     WM parameters for each blade node
  */

  NaluEnv::self().naluOutputP0()
    << "Setting Sample displacements " << std::endl;

  int nBlades = params_.numBlades;
  ;
  size_t nTotBldPts = 0;
  for (int i = 0; i < nBlades; i++)
    nTotBldPts += params_.nBRfsiPtsBlade[i];

  std::vector<double> x_axis = {1.0, 0.0, 0.0};
  std::vector<double> y_axis = {0.0, 1.0, 0.0};
  std::vector<double> z_axis = {0.0, 0.0, 1.0};

  // Step 1
  std::vector<double> hub_ref(3, 0.0);
  std::vector<double> hub_rot_axis(3, 0.0);
  for (int j = 0; j < 3; j++)
    hub_ref[j] = -brFSIdata_.hub_ref_pos[3 + j];
  applyWMrotation(hub_ref.data(), x_axis.data(), hub_rot_axis.data());

  // Turbine rotates at 12.1 rpm
  double omega = (9.156 / 60.0) * 2.0 * M_PI; // 12.1 rpm
  double theta = omega * curTime;

  double hub_rot = 4.0 * tan(0.25 * theta);
  std::vector<double> wm_hub_rot(3, 0.0);
  for (int j = 0; j < 3; j++) {
    wm_hub_rot[j] = hub_rot * hub_rot_axis[j];
    brFSIdata_.hub_def[3 + j] = -wm_hub_rot[j];
  }

  double sin_omegat = std::sin(omega * curTime);
  double pitch_rot = 4.0 * tan(0.25 * (0.0 * M_PI / 180.0) * sin_omegat);

  int istart = 0;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {

    brFSIdata_.bld_pitch[iBlade] = 0.0 * sin_omegat;

    // Step 2
    std::vector<double> pitch_axis_ref(3, 0.0);
    std::vector<double> wm_bld_root_ref(3, 0.0);
    for (int j = 0; j < 3; j++) {
      wm_bld_root_ref[j] = -brFSIdata_.bld_root_ref_pos[iBlade * 6 + 3 + j];
      brFSIdata_.bld_root_def[iBlade * 6 + 3 + j] =
        brFSIdata_.bld_root_ref_pos[iBlade * 6 + 3 + j];
    }

    applyWMrotation(
      wm_bld_root_ref.data(), z_axis.data(), pitch_axis_ref.data());

    std::vector<double> wm_pitch_rot(3, 0.0);
    for (int j = 0; j < 3; j++)
      wm_pitch_rot[j] = pitch_rot * pitch_axis_ref[j];

    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    for (int i = 0; i < nPtsBlade; i++) {

      double rloc = brFSIdata_.bld_rloc[istart + i];
      // double rdist_sq =
      // calcDistanceSquared(&(brFSIdata_.bld_ref_pos[(istart+i)*6]),
      // &(brFSIdata_.bld_ref_pos[(istart)*6]) )/10000.0;
      double rdist_sq = rloc * rloc;
      double sin_rdist_sq;
      if (rloc > 3.0)
        sin_rdist_sq =
          std::sin(rdist_sq) * 0.5 * (1 + std::tanh(0.8 * (rloc - 5.0)));
      else
        sin_rdist_sq = 0.0;

      // Step 3 - Set local rotational displacement
      std::vector<double> wm_loc_ref_node(3, 0.0);
      for (int j = 0; j < 3; j++)
        wm_loc_ref_node[j] = -brFSIdata_.bld_ref_pos[(istart + i) * 6 + 3 + j];

      std::vector<double> ref_loc_rot_axis = {
        1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
      std::vector<double> loc_rot_axis(3, 0.0);
      applyWMrotation(
        wm_loc_ref_node.data(), ref_loc_rot_axis.data(), loc_rot_axis.data());

      double loc_rot =
        4.0 * tan(0.25 * (0.0 * M_PI / 180.0) * sin_rdist_sq * sin_omegat);
      std::vector<double> wm_loc_rot(3, 0.0);

      for (int j = 0; j < 3; j++)
        wm_loc_rot[j] = loc_rot * loc_rot_axis[j];

      // Step 4 - Compose all rotations
      // Hub rotation, local deformation, pitch, reference orientation
      std::vector<double> tmp(3, 0.0);
      std::vector<double> tmp1(3, 0.0);
      composeWM(wm_loc_ref_node.data(), wm_pitch_rot.data(), tmp.data());
      // composeWM(wm_loc_rot.data(), tmp.data(), tmp1.data());
      // composeWM(wm_hub_rot.data(), tmp1.data(), tmp.data());

      for (int j = 0; j < 3; j++)
        brFSIdata_.bld_def[(istart + i) * 6 + 3 + j] = -tmp[j];

      // Step 5 - Set translational displacement
      double xDisp = 0.0; // sin_rdist_sq * 15.0 * sin_omegat;

      std::vector<double> r(3, 0.0);
      for (int j = 0; j < 3; j++)
        r[j] = brFSIdata_.bld_ref_pos[(istart + i) * 6 + j] -
               brFSIdata_.hub_ref_pos[j];

      std::vector<double> r_rot(3, 0.0);

      std::vector<double> trans_disp = {xDisp, xDisp, xDisp};
      std::vector<double> trans_disp_rot(3, 0.0);

      applyWMrotation(tmp.data(), trans_disp.data(), trans_disp_rot.data());

      applyWMrotation(wm_hub_rot.data(), r.data(), r_rot.data());

      brFSIdata_.bld_def[(istart + i) * 6 + 0] =
        trans_disp_rot[0] + r_rot[0] - r[0];
      brFSIdata_.bld_def[(istart + i) * 6 + 1] =
        trans_disp_rot[1] + r_rot[1] - r[1];
      brFSIdata_.bld_def[(istart + i) * 6 + 2] =
        trans_disp_rot[2] + r_rot[2] - r[2];
    }
    istart += nPtsBlade;
  }

  if (bulk_->parallel_rank() == 0) {
    std::ofstream bld_bm_mesh;
    bld_bm_mesh.open("blade_beam_mesh_setsample.csv", std::ios_base::out);
    for (int k = 0; k < params_.nBRfsiPtsBlade[0] * 3; k++) {
      bld_bm_mesh << brFSIdata_.bld_ref_pos[k * 6] << ","
                  << brFSIdata_.bld_ref_pos[k * 6 + 1] << ","
                  << brFSIdata_.bld_ref_pos[k * 6 + 2] << ","
                  << brFSIdata_.bld_def[k * 6] << ","
                  << brFSIdata_.bld_def[k * 6 + 1] << ","
                  << brFSIdata_.bld_def[k * 6 + 2] << ","
                  << brFSIdata_.bld_def[k * 6 + 3] << ","
                  << brFSIdata_.bld_def[k * 6 + 4] << ","
                  << brFSIdata_.bld_def[k * 6 + 5] << ","
                  << brFSIdata_.bld_vel[k * 6] << ","
                  << brFSIdata_.bld_vel[k * 6 + 1] << ","
                  << brFSIdata_.bld_vel[k * 6 + 2] << ","
                  << brFSIdata_.bld_vel[k * 6 + 3] << ","
                  << brFSIdata_.bld_vel[k * 6 + 4] << ","
                  << brFSIdata_.bld_vel[k * 6 + 5] << std::endl;
    }

    bld_bm_mesh.close();
  }
}

//! Set reference displacement on the turbine blade surface mesh, for comparison
//! with Sample displacement set in setSampleDisplacement
void
fsiTurbine::setRefDisplacement(double curTime)
{

  // Turbine rotates at 12.1 rpm
  double omega = (12.1 / 60.0) * 2.0 * M_PI; // 12.1 rpm

  // Rotate the hub first
  std::vector<double> hubPos = {0, 0, 0.0};
  std::vector<double> wmHubRot = {4.0 * tan(0.25 * 0.0), 0.0, 0.0};

  double sin_omegat = std::sin(omega * curTime);

  // extract the vector field type set by this function
  auto& meta = bulk_->mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  VectorFieldType* modelCoords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* refDisp = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement_ref");
  VectorFieldType* refVel = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_velocity_ref");

  std::vector<double> zAxis0 = {0.0, 0.0, 1.0};

  for (size_t iBlade = 0; iBlade < 3; iBlade++) {

    std::vector<double> wmRotBlade_ref = {
      4.0 * tan(0.25 * iBlade * 120.0 * M_PI / 180.0), 0.0, 0.0};
    std::vector<double> wmRotBlade(3, 0.0);
    composeWM(wmHubRot.data(), wmRotBlade_ref.data(), wmRotBlade.data());

    std::vector<double> nHatRef(3, 0.0);
    std::vector<double> nHat(3, 0.0);

    applyWMrotation(wmRotBlade_ref.data(), zAxis0.data(), nHatRef.data());
    applyWMrotation(wmRotBlade.data(), zAxis0.data(), nHat.data());

    stk::mesh::Selector sel(
      stk::mesh::selectUnion(bladeParts_[iBlade])); // extract blade
    const auto& bkts = bulk_->get_buckets(
      stk::topology::NODE_RANK, sel); // extract buckets for the blade

    for (auto b : bkts) { // loop over number of buckets
      for (size_t in = 0; in < b->size();
           in++) { // loop over all nodes in the bucket
        auto node = (*b)[in];
        double* xyz = stk::mesh::field_data(*modelCoords, node);
        double* vecRefNode = stk::mesh::field_data(*refDisp, node);
        double* velRefNode = stk::mesh::field_data(*refVel, node);

        std::vector<double> xyzMhub(3, 0.0);
        for (size_t j = 0; j < 3; j++)
          xyzMhub[j] = xyz[j] - hubPos[j];

        // Translational displacement due to turbine rotation
        std::vector<double> xyzMhubRot(3, 0.0);
        applyWMrotation(wmHubRot.data(), xyzMhub.data(), xyzMhubRot.data());

        // Compute position of current node relative to blade root
        double rDistSq = (dot(xyzMhub.data(), nHatRef.data()) - 1.5) *
                         (dot(xyzMhub.data(), nHatRef.data()) - 1.5) / 10000.0;
        double sinRdistSq = std::sin(rDistSq);
        double tanRdistSq = std::tan(rDistSq);

        // Set translational displacement due to deflection
        double xDisp = sinRdistSq * 15.0 * sin_omegat;
        std::vector<double> transDisp = {xDisp, xDisp, xDisp};
        std::vector<double> transDispRot(3, 0.0);
        applyWMrotation(
          wmRotBlade.data(), transDisp.data(), transDispRot.data());

        // Translational displacement due to rotational deflection
        double xyzMhubDotNHatRef = dot(xyzMhub.data(), nHatRef.data());
        std::vector<double> pGlobRef(3, 0.0);
        std::vector<double> pLoc(3, 0.0);
        for (int j = 0; j < 3; j++)
          pGlobRef[j] = xyzMhub[j] - xyzMhubDotNHatRef * nHatRef[j];
        applyWMrotation(
          wmRotBlade_ref.data(), pGlobRef.data(), pLoc.data(), -1.0);
        std::vector<double> pGlob(3, 0.0);
        applyWMrotation(wmRotBlade.data(), pLoc.data(), pGlob.data());

        double rot =
          4.0 *
          tan(
            0.25 * (0.0 * M_PI / 180.0) * sinRdistSq *
            sin_omegat); // 4.0 * tan(phi/4.0) parameter for Wiener-Milenkovic
        std::vector<double> wmRot1 = {
          1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
        std::vector<double> wmRot(3, 0.0);
        applyWMrotation(wmRotBlade.data(), wmRot1.data(), wmRot.data());
        for (int j = 0; j < 3; j++)
          wmRot[j] *= rot;

        std::vector<double> r_rot(3, 0.0);
        applyWMrotation(wmRot.data(), pGlob.data(), r_rot.data());

        for (int j = 0; j < ndim; j++)
          vecRefNode[j] =
            xyzMhubRot[j] - xyzMhub[j] + transDispRot[j] + r_rot[j] - pGlob[j];

        std::vector<double> omega = {
          sinRdistSq * 6.232, sinRdistSq * 6.232, sinRdistSq * 6.232};
        std::vector<double> omegaCrossRrot(3, 0.0);
        cross(omega.data(), r_rot.data(), omegaCrossRrot.data());
        for (int j = 0; j < ndim; j++)
          velRefNode[j] = tanRdistSq * 3.743 + omegaCrossRrot[j];
      }
    }
  }
}

//! Calculate the distance between 3-dimensional vectors 'a' and 'b'
double
fsiTurbine::calcDistanceSquared(double* a, double* b)
{

  double dist = 0.0;
  for (size_t i = 0; i < 3; i++)
    dist += (a[i] - b[i]) * (a[i] - b[i]);
  return dist;
}

//! Map the deflections from the openfast nodes to the turbine surface CFD mesh.
//! Will call 'computeDisplacement' for each node on the turbine surface CFD
//! mesh.
void
fsiTurbine::mapDisplacements(double time)
{

  // To implement this function - assume that 'dispMap_' field contains the
  // lower node id of the corresponding element along the blade or the tower
  // along with the 'bldDispMapInterp_' field that contains the non-dimensional
  // location of the CFD surface mesh node on that element.

  // For e.g., for blade 'k' if the lower node id from 'dispMap_' is 'j' and the
  // non-dimenional location from 'dispMapInterp_' is 'm', then the
  // translational displacement for the CFD surface mesh is (1-m) *
  // bld_def[k][j*6+0] + m * bld_def[k][(j+1)*6+0] (1-m) * bld_def[k][j*6+1] + m
  // * bld_def[k][(j+1)*6+1] (1-m) * bld_def[k][j*6+2] + m *
  // bld_def[k][(j+1)*6+2]

  const DeflectionRampingParams& defParams = deflectionRampParams_;
  const double temporalDeflectionRamp = fsi::temporal_ramp(
    time, defParams.startTimeTemporalRamp_, defParams.endTimeTemporalRamp_);

  auto& meta = bulk_->mesh_meta_data();
  VectorFieldType* modelCoords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* displacement = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "mesh_displacement");

  VectorFieldType* meshVelocity =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "mesh_velocity");

  std::vector<double> totDispNode(
    6, 0.0); // Total displacement at any node in (transX, transY, transZ, rotX,
             // rotY, rotZ)
  std::vector<double> totVelNode(
    6, 0.0); // Total velocity at any node in (transX, transY, transZ, rotX,
             // rotY, rotZ)
  std::vector<double> tmpNodePos(
    6, 0.0); // Vector to temporarily store a position and orientation vector

  // Do the tower first
  stk::mesh::Selector sel(stk::mesh::selectUnion(twrParts_));
  const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      auto oldxyz = vector_from_field(*modelCoords, node);
      int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
      const int iN = 6 * (*dispMapNode);
      const int iNp1 = 6 * (*dispMapNode + 1);
      double* dispMapInterpNode = stk::mesh::field_data(*dispMapInterp_, node);

      // Find the interpolated reference position first
      const auto twrStartRef = aero::SixDOF(&brFSIdata_.twr_ref_pos[iN]);
      const auto twrEndRef = aero::SixDOF(&brFSIdata_.twr_ref_pos[iNp1]);
      const auto refPos = aero::linear_interp_total_displacement(
        twrStartRef, twrEndRef, *dispMapInterpNode);

      // Now linearly interpolate the deflections to the intermediate location
      const auto twrStartDisp = aero::SixDOF(&brFSIdata_.twr_def[iN]);
      const auto twrEndDisp = aero::SixDOF(&brFSIdata_.twr_def[iNp1]);
      const auto deflection = aero::linear_interp_total_displacement(
                                twrStartDisp, twrEndDisp, *dispMapInterpNode) *
                              temporalDeflectionRamp;

      // Now transfer the interpolated displacement to the CFD mesh node
      vector_to_field(
        aero::compute_translational_displacements(deflection, refPos, oldxyz),
        *displacement, node);
    }
  }

  const aero::SixDOF hubVel(brFSIdata_.hub_vel.data());
  const aero::SixDOF hubDeflection(brFSIdata_.hub_def.data());
  const aero::SixDOF hubPos(brFSIdata_.hub_ref_pos.data());

  // Now the blades
  int nBlades = params_.numBlades;
  int iStart = 0;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    stk::mesh::Selector sel(stk::mesh::selectUnion(bladeParts_[iBlade]));
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b : bkts) {
      for (size_t in = 0; in < b->size(); in++) {
        auto node = (*b)[in];

        int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
        const int iN = 6 * (*dispMapNode + iStart);
        const int iNp1 = 6 * (*dispMapNode + iStart + 1);

        double* dispMapInterpNode =
          stk::mesh::field_data(*dispMapInterp_, node);

        // Find the interpolated reference position first
        auto bldStartRef = aero::SixDOF(&(brFSIdata_.bld_ref_pos[iN]));
        auto bldEndRef = aero::SixDOF(&(brFSIdata_.bld_ref_pos[iNp1]));
        auto refPos = aero::linear_interp_total_displacement(
          bldStartRef, bldEndRef, *dispMapInterpNode);

        // Now linearly interpolate the deflections to the intermediate
        auto bldStartDisp = aero::SixDOF(&(brFSIdata_.bld_def[iN]));
        auto bldEndDisp = aero::SixDOF(&(brFSIdata_.bld_def[iNp1]));
        auto interpDisp = aero::linear_interp_total_displacement(
          bldStartDisp, bldEndDisp, *dispMapInterpNode);

        // TODO(psakiev) ramping can be done entirely with reference coordinates
        // could cache this and do it once but might be better to do it inline
        // to save memory on gpus
        //
        // right now we do both (create field and compute inline) but it will be
        // easy to delete the field when this is no longer beta
        //
        // deflection ramping
        const double spanLocI = brFSIdata_.bld_rloc[*dispMapNode + iStart];
        const double spanLocIp1 =
          brFSIdata_.bld_rloc[*dispMapNode + iStart + 1];

        const double spanLocation =
          spanLocI + *dispMapInterpNode * (spanLocIp1 - spanLocI);

        double deflectionRamp =
          temporalDeflectionRamp *
          fsi::linear_ramp_span(spanLocation, defParams.spanRampDistance_);

        // things for theta mapping
        const aero::SixDOF hubPos(brFSIdata_.hub_ref_pos.data());
        const aero::SixDOF rootPos(&(brFSIdata_.bld_root_ref_pos[iBlade * 6]));
        const auto nodePosition = vector_from_field(*modelCoords, node);

        deflectionRamp *= fsi::linear_ramp_theta(
          hubPos, rootPos.position_, nodePosition, defParams.thetaRampSpan_,
          defParams.zeroRampLocTheta_);

        *stk::mesh::field_data(*deflectionRamp_, node) = deflectionRamp;

        // displacements from the hub will match a fully stiff blade's
        // displacements
        const auto hubBasedDef = aero::compute_translational_displacements(
          hubDeflection, hubPos, nodePosition);

        auto ramp_disp = aero::compute_translational_displacements(
          interpDisp, refPos, nodePosition, hubBasedDef, deflectionRamp);
        vector_to_field(ramp_disp, *displacement, node);

        auto bldStartVel = aero::SixDOF(&(brFSIdata_.bld_vel[iN]));
        auto bldEndVel = aero::SixDOF(&(brFSIdata_.bld_vel[iNp1]));
        auto interpVel = aero::linear_interp_total_velocity(
          bldStartVel, bldEndVel, *dispMapInterpNode);

        // Now transfer the translational and rotational velocity to an
        // equivalent translational velocity on the CFD mesh node
        const auto stiffVel = aero::compute_mesh_velocity(
          hubVel, hubDeflection, hubPos, nodePosition);

        vector_to_field(
          aero::compute_mesh_velocity(
            interpVel, interpDisp, refPos, nodePosition, stiffVel,
            deflectionRamp),
          *meshVelocity, node);
      }
    }
    iStart += nPtsBlade;
  }

  // Now the hub
  stk::mesh::Selector hubsel(stk::mesh::selectUnion(hubParts_));
  const auto& hubbkts = bulk_->get_buckets(stk::topology::NODE_RANK, hubsel);
  for (auto b : hubbkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];

      auto oldxyz = vector_from_field(*modelCoords, node);
      // Now transfer the displacement to the CFD mesh node
      vector_to_field(
        aero::compute_translational_displacements(
          hubDeflection, hubPos, oldxyz),
        *displacement, node);

      // Now transfer the translational and rotational velocity to an equivalent
      // translational velocity on the CFD mesh node
      vector_to_field(
        aero::compute_mesh_velocity(hubVel, hubDeflection, hubPos, oldxyz),
        *meshVelocity, node);
    }
  }

  // Now the nacelle
  stk::mesh::Selector nacelle(stk::mesh::selectUnion(nacelleParts_));
  const auto& nacbkts = bulk_->get_buckets(stk::topology::NODE_RANK, nacelle);
  for (auto b : nacbkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      auto oldxyz = vector_from_field(*modelCoords, node);
      const aero::SixDOF refPos(brFSIdata_.nac_ref_pos.data());
      const aero::SixDOF deflection(brFSIdata_.nac_def.data());
      // Now transfer the displacement to the CFD mesh node
      vector_to_field(
        aero::compute_translational_displacements(deflection, refPos, oldxyz),
        *displacement, node);

      // Now transfer the translational and rotational velocity to an equivalent
      // translational velocity on the CFD mesh node
      auto mVel = vector_from_field(*meshVelocity, node);
      const aero::SixDOF vel(brFSIdata_.nac_vel.data());

      mVel = aero::compute_mesh_velocity(vel, deflection, refPos, oldxyz);
    }
  }
}

//! Compose Wiener-Milenkovic parameters 'p' and 'q' into 'pPlusq'. If a
//! transpose of 'p' is required, set tranposeP to '-1', else leave blank or set
//! to '+1'
void
fsiTurbine::composeWM(
  double* p, double* q, double* pPlusq, double transposeP, double transposeQ)
{

  double p0 = 2.0 - 0.125 * dot(p, p);
  double q0 = 2.0 - 0.125 * dot(q, q);
  std::vector<double> pCrossq(3, 0.0);
  cross(p, q, pCrossq.data());

  double delta1 = (4.0 - p0) * (4.0 - q0);
  double delta2 = p0 * q0 - transposeP * transposeQ * dot(p, q);
  double premultFac = 0.0;
  if (delta2 < 0)
    premultFac = -4.0 / (delta1 - delta2);
  else
    premultFac = 4.0 / (delta1 + delta2);

  for (size_t i = 0; i < 3; i++)
    pPlusq[i] = premultFac * (transposeQ * p0 * q[i] + transposeP * q0 * p[i] +
                              transposeP * transposeQ * pCrossq[i]);
}

double
fsiTurbine::dot(double* a, double* b)
{

  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

void
fsiTurbine::cross(double* a, double* b, double* aCrossb)
{

  aCrossb[0] = a[1] * b[2] - a[2] * b[1];
  aCrossb[1] = a[2] * b[0] - a[0] * b[2];
  aCrossb[2] = a[0] * b[1] - a[1] * b[0];
}

//! Apply a Wiener-Milenkovic rotation 'wm' to a vector 'r' into 'rRot'. To
//! optionally transpose the rotation, set 'tranpose=-1.0'.
void
fsiTurbine::applyWMrotation(
  double* wm, double* r, double* rRot, double transpose)
{

  double wm0 = 2.0 - 0.125 * dot(wm, wm);
  double nu = 2.0 / (4.0 - wm0);
  double cosPhiO2 = 0.5 * wm0 * nu;
  std::vector<double> wmCrossR(3, 0.0);
  cross(wm, r, wmCrossR.data());
  std::vector<double> wmCrosswmCrossR(3, 0.0);
  cross(wm, wmCrossR.data(), wmCrosswmCrossR.data());

  for (size_t i = 0; i < 3; i++)
    rRot[i] = r[i] + transpose * nu * cosPhiO2 * wmCrossR[i] +
              0.5 * nu * nu * wmCrosswmCrossR[i];
}

//! Map each node on the turbine surface CFD mesh to the blade beam mesh
void
fsiTurbine::computeMapping()
{

  auto& meta = bulk_->mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  ThrowRequireMsg(ndim == 3, "fsiTurbine: spatial dim is required to be 3.");
  VectorFieldType* modelCoords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  // Do the tower first
  stk::mesh::Selector sel(stk::mesh::selectUnion(twrParts_));
  const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      double* xyz = stk::mesh::field_data(*modelCoords, node);
      int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
      double* dispMapInterpNode = stk::mesh::field_data(*dispMapInterp_, node);
      vs::Vector ptCoords(xyz[0], xyz[1], xyz[2]);
      bool foundProj = false;
      double nDimCoord = -1.0;
      int nPtsTwr = params_.nBRfsiPtsTwr;
      if (nPtsTwr > 0) {
        for (int i = 0; i < nPtsTwr - 1; i++) {
          vs::Vector lStart = {
            brFSIdata_.twr_ref_pos[i * 6], brFSIdata_.twr_ref_pos[i * 6 + 1],
            brFSIdata_.twr_ref_pos[i * 6 + 2]};
          vs::Vector lEnd = {
            brFSIdata_.twr_ref_pos[(i + 1) * 6],
            brFSIdata_.twr_ref_pos[(i + 1) * 6 + 1],
            brFSIdata_.twr_ref_pos[(i + 1) * 6 + 2]};
          nDimCoord = fsi::projectPt2Line(ptCoords, lStart, lEnd);

          if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
            *dispMapInterpNode = nDimCoord;
            *dispMapNode = i;
            //                        *loadMapNode = i + std::round(nDimCoord);
            foundProj = true;
            break;
          }
        }

        // If no element in the OpenFAST mesh contains this node do some sanity
        // check on the perpendicular distance between the surface mesh node and
        // the line joining the ends of the tower
        if (!foundProj) {
          vs::Vector lStart = {
            brFSIdata_.twr_ref_pos[0], brFSIdata_.twr_ref_pos[1],
            brFSIdata_.twr_ref_pos[2]};
          vs::Vector lEnd = {
            brFSIdata_.twr_ref_pos[(nPtsTwr - 1) * 6],
            brFSIdata_.twr_ref_pos[(nPtsTwr - 1) * 6 + 1],
            brFSIdata_.twr_ref_pos[(nPtsTwr - 1) * 6 + 2]};
          double perpDist =
            fsi::perpProjectDist_Pt2Line(ptCoords, lStart, lEnd);
          if (perpDist > 1.0) { // Something's wrong if a node on the surface
                                // mesh of the tower is more than 20% of the
                                // tower length away from the tower axis.
            throw std::runtime_error(
              "Can't find a projection for point (" +
              std::to_string(ptCoords[0]) + "," + std::to_string(ptCoords[1]) +
              "," + std::to_string(ptCoords[2]) + ") on the tower on turbine " +
              std::to_string(params_.TurbID) + ". The tower extends from " +
              std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) +
              "," + std::to_string(lStart[2]) + ") to " +
              std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," +
              std::to_string(lEnd[2]) +
              "). Are you sure the initial position and orientation of the "
              "mesh is consistent with the input file parameters and the "
              "OpenFAST model.");
          }
          if (nDimCoord < 0.0) {
            // Assign this node to the first point and element of the OpenFAST
            // mesh
            *dispMapInterpNode = 0.0;
            *dispMapNode = 0;
            //                        *loadMapNode = 0;
          } else if (nDimCoord > 1.0) { // Assign this node to the last point
                                        // and element of the OpenFAST mesh
            *dispMapInterpNode = 1.0;
            *dispMapNode = nPtsTwr - 2;
            //                        *loadMapNode = nPtsTwr-1;
          }
        }
      }
    }
  }

  // Now the blades
  int nBlades = params_.numBlades;
  int iStart = 0;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    stk::mesh::Selector sel(stk::mesh::selectUnion(bladeParts_[iBlade]));
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

    for (auto b : bkts) {
      for (size_t in = 0; in < b->size(); in++) {
        auto node = (*b)[in];
        double* xyz = stk::mesh::field_data(*modelCoords, node);
        int* dispMapNode = stk::mesh::field_data(*dispMap_, node);
        double* dispMapInterpNode =
          stk::mesh::field_data(*dispMapInterp_, node);
        vs::Vector ptCoords(xyz[0], xyz[1], xyz[2]);
        bool foundProj = false;
        double nDimCoord = -1.0;
        for (int i = 0; i < nPtsBlade - 1; i++) {
          vs::Vector lStart = {
            brFSIdata_.bld_ref_pos[(iStart + i) * 6],
            brFSIdata_.bld_ref_pos[(iStart + i) * 6 + 1],
            brFSIdata_.bld_ref_pos[(iStart + i) * 6 + 2]};
          vs::Vector lEnd = {
            brFSIdata_.bld_ref_pos[(iStart + i + 1) * 6],
            brFSIdata_.bld_ref_pos[(iStart + i + 1) * 6 + 1],
            brFSIdata_.bld_ref_pos[(iStart + i + 1) * 6 + 2]};
          nDimCoord = fsi::projectPt2Line(ptCoords, lStart, lEnd);

          if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
            foundProj = true;
            *dispMapInterpNode = nDimCoord;
            *dispMapNode = i;
            //                        *loadMapNode = i + std::round(nDimCoord);
            break;
          }
        }

        // If no element in the OpenFAST mesh contains this node do some sanity
        // check on the perpendicular distance between the surface mesh node and
        // the line joining the ends of the blade
        if (!foundProj) {

          std::vector<double> lStart = {
            brFSIdata_.bld_ref_pos[iStart * 6],
            brFSIdata_.bld_ref_pos[iStart * 6 + 1],
            brFSIdata_.bld_ref_pos[iStart * 6 + 2]};
          std::vector<double> lEnd = {
            brFSIdata_.bld_ref_pos[(iStart + nPtsBlade - 1) * 6],
            brFSIdata_.bld_ref_pos[(iStart + nPtsBlade - 1) * 6 + 1],
            brFSIdata_.bld_ref_pos[(iStart + nPtsBlade - 1) * 6 + 2]};

          if (nDimCoord < 0.0) {
            // Assign this node to the first point and element of the OpenFAST
            // mesh
            *dispMapInterpNode = 0.0;
            *dispMapNode = 0;
            //                            *loadMapNode = 0;
          } else if (nDimCoord > 1.0) { // Assign this node to the last point
                                        // and element of the OpenFAST mesh
            *dispMapInterpNode = 1.0;
            *dispMapNode = nPtsBlade - 2;
            //                        *loadMapNode = nPtsBlade-1;
          }
        }
      }
    }
    iStart += nPtsBlade;
  }

  // Write reference positions to netcdf file
  // write_nc_ref_pos();
}

//! Map each sub-control surface on the turbine surface CFD mesh to the blade
//! beam mesh
void
fsiTurbine::computeLoadMapping()
{

  auto& meta = bulk_->mesh_meta_data();
  const int ndim = meta.spatial_dimension();
  VectorFieldType* modelCoords =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "coordinates");

  // nodal fields to gather
  std::vector<double> ws_coordinates;
  vs::Vector coord_bip(0.0, 0.0, 0.0);
  std::vector<double> ws_face_shape_function;

  // Do the tower first
  stk::mesh::Selector sel(
    meta.locally_owned_part() & stk::mesh::selectUnion(twrBndyParts_));
  const auto& bkts = bulk_->get_buckets(meta.side_rank(), sel);

  for (auto b : bkts) {
    // face master element
    MasterElement* meFC =
      MasterElementRepo::get_surface_master_element_on_host(b->topology());
    const int nodesPerFace = meFC->nodesPerElement_;
    const int numScsBip = meFC->num_integration_points();

    // mapping from ip to nodes for this ordinal;
    // face perspective (use with face_node_relations)
    ws_face_shape_function.resize(numScsBip * nodesPerFace);

    SharedMemView<double**, HostShmem> p_face_shape_function(
      ws_face_shape_function.data(), numScsBip, nodesPerFace);

    meFC->shape_fcn<>(p_face_shape_function);

    ws_coordinates.resize(ndim * nodesPerFace);

    for (size_t in = 0; in < b->size(); in++) {

      // get face
      stk::mesh::Entity face = (*b)[in];
      // face node relations
      stk::mesh::Entity const* face_node_rels = bulk_->begin_nodes(face);
      // gather nodal data off of face
      for (int ni = 0; ni < nodesPerFace; ++ni) {
        stk::mesh::Entity node = face_node_rels[ni];
        // gather coordinates
        const double* xyz = stk::mesh::field_data(*modelCoords, node);
        for (auto i = 0; i < ndim; i++)
          ws_coordinates[ni * ndim + i] = xyz[i];
      }

      // Get reference to load map and loadMapInterp at all ips on this face
      int* loadMapFace = stk::mesh::field_data(*loadMap_, face);
      double* loadMapInterpFace = stk::mesh::field_data(*loadMapInterp_, face);

      for (int ip = 0; ip < numScsBip; ++ip) {
        // Get coordinates of this ip
        for (auto i = 0; i < ndim; i++)
          coord_bip[i] = 0.0;
        for (int ni = 0; ni < nodesPerFace; ni++) {
          for (int i = 0; i < ndim; i++)
            coord_bip[i] +=
              p_face_shape_function(ip, ni) * ws_coordinates[ni * ndim + i];
        }

        // Create map at this ip
        bool foundProj = false;
        double nDimCoord = -1.0;
        int nPtsTwr = params_.nBRfsiPtsTwr;
        if (nPtsTwr > 0) {
          for (int i = 0; i < nPtsTwr - 1; i++) {
            vs::Vector lStart = {
              brFSIdata_.twr_ref_pos[i * 6], brFSIdata_.twr_ref_pos[i * 6 + 1],
              brFSIdata_.twr_ref_pos[i * 6 + 2]};
            vs::Vector lEnd = {
              brFSIdata_.twr_ref_pos[(i + 1) * 6],
              brFSIdata_.twr_ref_pos[(i + 1) * 6 + 1],
              brFSIdata_.twr_ref_pos[(i + 1) * 6 + 2]};
            nDimCoord = fsi::projectPt2Line(coord_bip, lStart, lEnd);

            if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
              loadMapInterpFace[ip] = nDimCoord;
              loadMapFace[ip] = i;
              foundProj = true;
              break;
            }
          }

          // If no element in the OpenFAST mesh contains this node do
          // some sanity check on the perpendicular distance between
          // the surface mesh node and the line joining the ends of the
          // tower
          if (!foundProj) {
            vs::Vector lStart = {
              brFSIdata_.twr_ref_pos[0], brFSIdata_.twr_ref_pos[1],
              brFSIdata_.twr_ref_pos[2]};
            vs::Vector lEnd = {
              brFSIdata_.twr_ref_pos[(nPtsTwr - 1) * 6],
              brFSIdata_.twr_ref_pos[(nPtsTwr - 1) * 6 + 1],
              brFSIdata_.twr_ref_pos[(nPtsTwr - 1) * 6 + 2]};
            double perpDist =
              fsi::perpProjectDist_Pt2Line(coord_bip, lStart, lEnd);
            // Something's wrong if a node on the surface mesh of
            // the tower is more than 20% of the tower length away
            // from the tower axis.
            if (perpDist > 1.0) {
              throw std::runtime_error(
                "Can't find a projection for point (" +
                std::to_string(coord_bip[0]) + "," +
                std::to_string(coord_bip[1]) + "," +
                std::to_string(coord_bip[2]) + ") on the tower on turbine " +
                std::to_string(params_.TurbID) + ". The tower extends from " +
                std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) +
                "," + std::to_string(lStart[2]) + ") to " +
                std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," +
                std::to_string(lEnd[2]) +
                "). Are you sure the initial position and orientation of the "
                "mesh is consistent with the input file parameters and the "
                "OpenFAST model.");
            }
            if (nDimCoord < 0.0) {
              // Assign this node to the first point and
              // element of the OpenFAST mesh
              loadMapInterpFace[ip] = 0.0;
              loadMapFace[ip] = 0;
            } else if (nDimCoord > 1.0) {
              // Assign this node to the last point and
              // element of the OpenFAST mesh
              loadMapInterpFace[ip] = 1.0;
              loadMapFace[ip] = nPtsTwr - 2;
            }
          }
        }
      }
    }
  }

  // Now the blades
  int nBlades = params_.numBlades;
  int iStart = 0;
  for (int iBlade = 0; iBlade < nBlades; iBlade++) {
    std::vector<double> cfd_mesh_rloc;
    int nPtsBlade = params_.nBRfsiPtsBlade[iBlade];
    stk::mesh::Selector sel(
      meta.locally_owned_part() &
      stk::mesh::selectUnion(bladeBndyParts_[iBlade]));
    const auto& bkts = bulk_->get_buckets(meta.side_rank(), sel);

    for (auto b : bkts) {
      // face master element
      MasterElement* meFC =
        MasterElementRepo::get_surface_master_element_on_host(b->topology());
      const int nodesPerFace = meFC->nodesPerElement_;
      const int numScsBip = meFC->num_integration_points();

      // mapping from ip to nodes for this ordinal;
      // face perspective (use with face_node_relations)
      ws_face_shape_function.resize(numScsBip * nodesPerFace);

      SharedMemView<double**, HostShmem> p_face_shape_function(
        ws_face_shape_function.data(), numScsBip, nodesPerFace);

      meFC->shape_fcn<>(p_face_shape_function);

      ws_coordinates.resize(ndim * nodesPerFace);

      for (size_t in = 0; in < b->size(); in++) {

        // get face
        stk::mesh::Entity face = (*b)[in];
        // face node relations
        stk::mesh::Entity const* face_node_rels = bulk_->begin_nodes(face);
        // gather nodal data off of face
        for (int ni = 0; ni < nodesPerFace; ++ni) {
          stk::mesh::Entity node = face_node_rels[ni];
          // gather coordinates
          const double* xyz = stk::mesh::field_data(*modelCoords, node);
          for (auto i = 0; i < ndim; i++)
            ws_coordinates[ni * ndim + i] = xyz[i];
        }

        // Get reference to load map and loadMapInterp at all ips on this face
        int* loadMapFace = stk::mesh::field_data(*loadMap_, face);
        double* loadMapInterpFace =
          stk::mesh::field_data(*loadMapInterp_, face);

        for (int ip = 0; ip < numScsBip; ++ip) {
          // Get coordinates of this ip
          for (auto i = 0; i < ndim; i++)
            coord_bip[i] = 0.0;
          for (int ni = 0; ni < nodesPerFace; ni++) {
            for (int i = 0; i < ndim; i++)
              coord_bip[i] +=
                p_face_shape_function(ip, ni) * ws_coordinates[ni * ndim + i];
          }

          bool foundProj = false;
          double nDimCoord = -1.0;
          for (int i = 0; i < nPtsBlade - 1; i++) {
            vs::Vector lStart = {
              brFSIdata_.bld_ref_pos[(iStart + i) * 6],
              brFSIdata_.bld_ref_pos[(iStart + i) * 6 + 1],
              brFSIdata_.bld_ref_pos[(iStart + i) * 6 + 2]};
            vs::Vector lEnd = {
              brFSIdata_.bld_ref_pos[(iStart + i + 1) * 6],
              brFSIdata_.bld_ref_pos[(iStart + i + 1) * 6 + 1],
              brFSIdata_.bld_ref_pos[(iStart + i + 1) * 6 + 2]};
            nDimCoord = fsi::projectPt2Line(coord_bip, lStart, lEnd);
            if ((nDimCoord >= 0) && (nDimCoord <= 1.0)) {
              foundProj = true;
              loadMapInterpFace[ip] = nDimCoord;
              loadMapFace[ip] = i;
              break;
            }
          }

          // If no element in the OpenFAST mesh contains this
          // node do some sanity check on the perpendicular
          // distance between the surface mesh node and the line
          // joining the ends of the blade
          if (!foundProj) {

            vs::Vector lStart = {
              brFSIdata_.bld_ref_pos[iStart * 6],
              brFSIdata_.bld_ref_pos[iStart * 6 + 1],
              brFSIdata_.bld_ref_pos[iStart * 6 + 2]};
            vs::Vector lEnd = {
              brFSIdata_.bld_ref_pos[(iStart + nPtsBlade - 1) * 6],
              brFSIdata_.bld_ref_pos[(iStart + nPtsBlade - 1) * 6 + 1],
              brFSIdata_.bld_ref_pos[(iStart + nPtsBlade - 1) * 6 + 2]};

            double perpDist =
              fsi::perpProjectDist_Pt2Line(coord_bip, lStart, lEnd);
            // Something's wrong if a node on the surface
            // mesh of the blade is more than 20% of the
            // blade length away from the blade axis.
            if (perpDist > 1.0) {
              throw std::runtime_error(
                "Can't find a projection for point (" +
                std::to_string(coord_bip[0]) + "," +
                std::to_string(coord_bip[1]) + "," +
                std::to_string(coord_bip[2]) + ") on blade " +
                std::to_string(iBlade) + " on turbine " +
                std::to_string(params_.TurbID) + ". The blade extends from " +
                std::to_string(lStart[0]) + "," + std::to_string(lStart[1]) +
                "," + std::to_string(lStart[2]) + ") to " +
                std::to_string(lEnd[0]) + "," + std::to_string(lEnd[1]) + "," +
                std::to_string(lEnd[2]) +
                "). Are you sure the initial position and orientation of the "
                "mesh is consistent with the input file parameters and the "
                "OpenFAST model.");
            }

            if (nDimCoord < 0.0) {
              // Assign this node to the first point and element of the OpenFAST
              // mesh
              loadMapInterpFace[ip] = 0.0;
              loadMapFace[ip] = 0;
            } else if (nDimCoord > 1.0) { // Assign this node to the last point
                                          // and element of the OpenFAST mesh
              loadMapInterpFace[ip] =
                1.0; // brFSIdata_.bld_rloc[iStart+nPtsBlade-1];
              loadMapFace[ip] = nPtsBlade - 2;
            }
          }
        }
      }
    }

    iStart += nPtsBlade;
  }
}

void
fsiTurbine::compute_div_mesh_velocity()
{

  auto& meta = bulk_->mesh_meta_data();

  ScalarFieldType* divMeshVel = meta.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "div_mesh_velocity");

  GenericFieldType* faceVelMag = meta.get_field<GenericFieldType>(
    stk::topology::EDGE_RANK, "edge_face_velocity_mag");

  compute_edge_scalar_divergence(
    *bulk_, partVec_, bndyPartVec_, faceVelMag, divMeshVel);
}

} // namespace nalu

} // namespace sierra
