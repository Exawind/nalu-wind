// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "aero/fsi/OpenfastFSI.h"
#include "aero/fsi/FSIturbine.h"
#include <NaluParsing.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace sierra {

namespace nalu {

OpenfastFSI::OpenfastFSI(const YAML::Node& node)
  : mesh_motion_(false), enable_calc_loads_(true)
{
  load(node);
}

void
OpenfastFSI::end_openfast()
{
  FAST.end();
}

void
OpenfastFSI::read_turbine_data(
  int iTurb, fast::fastInputs& fi, YAML::Node turbNode)
{

  // Read turbine data for a given turbine using the YAML node
  if (turbNode["turb_id"]) {
    fi.globTurbineData[iTurb].TurbID = turbNode["turb_id"].as<int>();
  } else {
    turbNode["turb_id"] = iTurb;
  }
  if (turbNode["sim_type"]) {
    if (turbNode["sim_type"].as<std::string>() == "ext-loads") {
      fi.globTurbineData[iTurb].sType = fast::EXTLOADS;
      fsiTurbineData_[iTurb] = std::make_unique<fsiTurbine>(iTurb, turbNode);
      mesh_motion_ = true;
    } else {
      fi.globTurbineData[iTurb].sType = fast::EXTINFLOW;
    }
  } else {
    fi.globTurbineData[iTurb].sType = fast::EXTINFLOW;
  }
  if (turbNode["FAST_input_filename"]) {
    fi.globTurbineData[iTurb].FASTInputFileName =
      turbNode["FAST_input_filename"].as<std::string>();
  } else {
    fi.globTurbineData[iTurb].FASTInputFileName = "";
  }
  if (turbNode["restart_filename"]) {
    fi.globTurbineData[iTurb].FASTRestartFileName =
      turbNode["restart_filename"].as<std::string>();
  } else {
    fi.globTurbineData[iTurb].FASTRestartFileName = "";
  }
  if (
    (fi.globTurbineData[iTurb].FASTRestartFileName == "") &&
    (fi.globTurbineData[iTurb].FASTInputFileName == ""))
    throw std::runtime_error(
      "Both FAST_input_filename and restart_filename are empty or not "
      "specified for Turbine " +
      std::to_string(iTurb));
  if (turbNode["turbine_base_pos"].IsSequence()) {
    fi.globTurbineData[iTurb].TurbineBasePos =
      turbNode["turbine_base_pos"].as<std::vector<float>>();
  } else {
    fi.globTurbineData[iTurb].TurbineBasePos = std::vector<float>(3, 0.0);
  }
  if (turbNode["turbine_hub_pos"].IsSequence()) {
    fi.globTurbineData[iTurb].TurbineHubPos =
      turbNode["turbine_hub_pos"].as<std::vector<double>>();
  } else {
    fi.globTurbineData[iTurb].TurbineHubPos = std::vector<double>(3, 0.0);
  }
  if (turbNode["num_force_pts_blade"]) {
    fi.globTurbineData[iTurb].numForcePtsBlade =
      turbNode["num_force_pts_blade"].as<int>();
  } else {
    fi.globTurbineData[iTurb].numForcePtsBlade = 0;
  }
  if (turbNode["num_force_pts_tower"]) {
    fi.globTurbineData[iTurb].numForcePtsTwr =
      turbNode["num_force_pts_tower"].as<int>();
  } else {
    fi.globTurbineData[iTurb].numForcePtsTwr = 0;
  }
  if (turbNode["nacelle_cd"]) {
    fi.globTurbineData[iTurb].nacelle_cd = turbNode["nacelle_cd"].as<float>();
  } else {
    fi.globTurbineData[iTurb].nacelle_cd = 0.0;
  }
  if (turbNode["nacelle_area"]) {
    fi.globTurbineData[iTurb].nacelle_area =
      turbNode["nacelle_area"].as<float>();
  } else {
    fi.globTurbineData[iTurb].nacelle_area = 0.0;
  }
  if (turbNode["air_density"]) {
    fi.globTurbineData[iTurb].air_density = turbNode["air_density"].as<float>();
  } else {
    fi.globTurbineData[iTurb].air_density = 0.0;
  }
}

void
OpenfastFSI::load(const YAML::Node& node)
{

  fi.comm = NaluEnv::self().parallel_comm();

  get_required(node, "n_turbines_glob", fi.nTurbinesGlob);

  if (fi.nTurbinesGlob > 0) {

    get_if_present(node, "dry_run", fi.dryRun, false);
    get_if_present(node, "debug", fi.debug, false);
    std::string simStartType = "na";
    get_required(node, "sim_start", simStartType);
    if (simStartType == "init") {
      fi.simStart = fast::init;
    } else if (simStartType == "trueRestart") {
      fi.simStart = fast::trueRestart;
    } else if (simStartType == "restartDriverInitFAST") {
      fi.simStart = fast::restartDriverInitFAST;
    }
    get_required(node, "t_max", fi.tMax); // tMax is the total
    // duration to which you want to run FAST.  This should be the
    // same or greater than the max time given in the FAST fst
    // file. Choose this carefully as FAST writes the output file
    // only at this point if you choose the binary file output.

    if (node["super_controller"]) {
      get_required(node, "super_controller", fi.scStatus);
      get_required(node, "sc_libFile", fi.scLibFile);
      get_required(node, "num_sc_inputs", fi.numScInputs);
      get_required(node, "num_sc_outputs", fi.numScOutputs);
    }

    fsiTurbineData_.resize(fi.nTurbinesGlob);
    fi.globTurbineData.resize(fi.nTurbinesGlob);
    for (int iTurb = 0; iTurb < fi.nTurbinesGlob; iTurb++) {
      if (node["Turbine" + std::to_string(iTurb)]) {
        read_turbine_data(iTurb, fi, node["Turbine" + std::to_string(iTurb)]);
      } else {
        throw std::runtime_error(
          "Node for Turbine" + std::to_string(iTurb) +
          " not present in input file or I cannot read it");
      }
    }

  } else {
    throw std::runtime_error("Number of turbines <= 0 ");
  }

  FAST.setInputs(fi);
}

void
OpenfastFSI::setup(double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  dt_ = dtNalu;

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) // This may not be a turbine intended for
                                    // blade-resolved simulation
      fsiTurbineData_[i]->setup(bulk_);
  }
  FAST.allocateTurbinesToProcsSimple();
  FAST.setDriverTimeStep(dtNalu);
  FAST.init();
}

void
OpenfastFSI::initialize(int restartFreqNalu, double curTime)
{

  FAST.setDriverCheckpoint(restartFreqNalu);
  // TODO: Check here on the processor containing the turbine that the number of
  // blades on the turbine is the same as the number of blade parts specified in
  // the input file.

  // TODO: In the documentation, mention that the CFD mesh must always be
  // created for the turbine in the reference position defined in OpenFAST, i.e.
  // with blade1 pointing up and the other blades following it in order as the
  // turbine rotates clockwise facing downwind. If the mesh is not created this
  // way, the mapping won't work. Any non-zero initial azimuth and/or initial
  // yaw must be only specified in the OpenFAST input file and the mesh will
  // automatically be deformed after calling solution0. Requiring the initial
  // CFD mesh to be in the reference configuration may not always work if the
  // mesh domain and initial yaw setting does not align with the reference
  // configuration. May be this is isn't an issue because unlike AeroDyn,
  // ExtLoads does not create the initial mesh independent of the
  // ElastoDyn/BeamDyn. May be ExtLoads already has the correct yaw and azimuth
  // setting from OpenFAST after the init call. In this case, the CFD mesh must
  // start in the correct azimuth and yaw configuration. In which case, the
  // initial yaw and azimuth must be obtained from OpenFAST and the mesh around
  // the turbine must be deformed through rigid body motion first before
  // starting any mapping.

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    // This may not be a turbine intended for blade-resolved simulation
    if (fsiTurbineData_[i] != NULL) {
      int turbProc = FAST.getProc(i);
      fsiTurbineData_[i]->setProc(turbProc);
      if (bulk_->parallel_rank() == turbProc) {
        FAST.get_turbineParams(i, fsiTurbineData_[i]->params_);
      }
      bcast_turbine_params(i);
      fsiTurbineData_[i]->initialize();
    }
  }

  compute_mapping();

  if (FAST.isTimeZero()) {
    send_loads(0.0);
    FAST.solution0();
  }

  map_displacements(curTime, false);

  if (curTime < 1e-10) {

    NaluEnv::self().naluOutputP0()
      << "Setting displacements at time steps n and n-1" << std::endl;

    auto& meta = bulk_->mesh_meta_data();

    const VectorFieldType* meshDisp =
      meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");
    const VectorFieldType* meshVel =
      meta.get_field<double>(stk::topology::NODE_RANK, "mesh_velocity");

    const VectorFieldType* meshDispNp1 =
      &(meshDisp->field_of_state(stk::mesh::StateNP1));
    VectorFieldType* meshDispN = &(meshDisp->field_of_state(stk::mesh::StateN));
    VectorFieldType* meshDispNm1 =
      &(meshDisp->field_of_state(stk::mesh::StateNM1));
    const VectorFieldType* meshVelNp1 =
      &(meshVel->field_of_state(stk::mesh::StateNP1));

    meshDisp->sync_to_host();
    meshVel->sync_to_host();
    meshDispNp1->sync_to_host();
    meshDispN->sync_to_host();
    meshDispNm1->sync_to_host();
    meshVelNp1->sync_to_host();

    stk::mesh::Selector sel = meta.universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        const double* velNp1 = stk::mesh::field_data(*meshVelNp1, node);
        const double* dispNp1 = stk::mesh::field_data(*meshDispNp1, node);
        double* dispN = stk::mesh::field_data(*meshDispN, node);
        double* dispNm1 = stk::mesh::field_data(*meshDispNm1, node);
        for (size_t i = 0; i < 3; i++) {
          dispN[i] = dispNp1[i] - dt_ * velNp1[i];
          dispNm1[i] = dispNp1[i] - 2.0 * dt_ * velNp1[i];
        }
      }
    }
    meshDispN->modify_on_host();
    meshDispNm1->modify_on_host();
  }
}

void
OpenfastFSI::bcast_turbine_params(int iTurb)
{

  std::vector<int> tIntParams(7, 0); // Assumes a max number of blades of 3
  std::vector<double> tDoubleParams(6, 0.0);
  int turbProc = fsiTurbineData_[iTurb]->getProc();
  if (bulk_->parallel_rank() == turbProc) {
    tIntParams[0] = fsiTurbineData_[iTurb]->params_.TurbID;
    tIntParams[1] = fsiTurbineData_[iTurb]->params_.numBlades;
    tIntParams[2] = fsiTurbineData_[iTurb]->params_.nTotBRfsiPtsBlade;
    tIntParams[3] = fsiTurbineData_[iTurb]->params_.nBRfsiPtsTwr;
    for (int i = 0; i < fsiTurbineData_[iTurb]->params_.numBlades; i++) {
      tIntParams[4 + i] = fsiTurbineData_[iTurb]->params_.nBRfsiPtsBlade[i];
    }

    for (int i = 0; i < 3; i++) {
      tDoubleParams[i] = fsiTurbineData_[iTurb]->params_.TurbineBasePos[i];
      tDoubleParams[3 + i] = fsiTurbineData_[iTurb]->params_.TurbineHubPos[i];
    }
  }
  MPI_Bcast(tIntParams.data(), 7, MPI_INT, turbProc, bulk_->parallel());
  MPI_Bcast(tDoubleParams.data(), 6, MPI_DOUBLE, turbProc, bulk_->parallel());

  if (bulk_->parallel_rank() != turbProc) {
    fsiTurbineData_[iTurb]->params_.TurbID = tIntParams[0];
    fsiTurbineData_[iTurb]->params_.numBlades = tIntParams[1];
    fsiTurbineData_[iTurb]->params_.nTotBRfsiPtsBlade = tIntParams[2];
    fsiTurbineData_[iTurb]->params_.nBRfsiPtsTwr = tIntParams[3];
    fsiTurbineData_[iTurb]->params_.nBRfsiPtsBlade.resize(
      fsiTurbineData_[iTurb]->params_.numBlades);
    for (int i = 0; i < fsiTurbineData_[iTurb]->params_.numBlades; i++) {
      fsiTurbineData_[iTurb]->params_.nBRfsiPtsBlade[i] = tIntParams[4 + i];
    }

    fsiTurbineData_[iTurb]->params_.TurbineBasePos.resize(3);
    fsiTurbineData_[iTurb]->params_.TurbineHubPos.resize(3);
    for (int i = 0; i < 3; i++) {
      fsiTurbineData_[iTurb]->params_.TurbineBasePos[i] = tDoubleParams[i];
      fsiTurbineData_[iTurb]->params_.TurbineHubPos[i] = tDoubleParams[3 + i];
    }
  }
}

void
OpenfastFSI::compute_mapping()
{

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) { // This may not be a turbine intended for
                                      // blade-resolved simulation
      int turbProc = fsiTurbineData_[i]->getProc();
      if (bulk_->parallel_rank() == turbProc) {
        FAST.getTowerRefPositions(
          fsiTurbineData_[i]->brFSIdata_.twr_ref_pos.data(), i);
        FAST.getBladeRefPositions(
          fsiTurbineData_[i]->brFSIdata_.bld_ref_pos.data(), i);
        FAST.getBladeRootRefPositions(
          fsiTurbineData_[i]->brFSIdata_.bld_root_ref_pos.data(), i);
        FAST.getHubRefPosition(
          fsiTurbineData_[i]->brFSIdata_.hub_ref_pos.data(), i);
        FAST.getNacelleRefPosition(
          fsiTurbineData_[i]->brFSIdata_.nac_ref_pos.data(), i);
        FAST.getBladeRloc(fsiTurbineData_[i]->brFSIdata_.bld_rloc.data(), i);
        FAST.getBladeChord(fsiTurbineData_[i]->brFSIdata_.bld_chord.data(), i);
      }

      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.twr_ref_pos.data(),
        (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());
      int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
      int nBlades = fsiTurbineData_[i]->params_.numBlades;
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_ref_pos.data(), nTotBldNodes * 6,
        MPI_DOUBLE, turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_root_ref_pos.data(), nBlades * 6,
        MPI_DOUBLE, turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.hub_ref_pos.data(), 6, MPI_DOUBLE,
        turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.nac_ref_pos.data(), 6, MPI_DOUBLE,
        turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_rloc.data(), nTotBldNodes,
        MPI_DOUBLE, turbProc, bulk_->parallel());
      // No need to bcast chord
      fsiTurbineData_[i]->computeMapping();
      fsiTurbineData_[i]->computeLoadMapping();
    }
  }
}

void
OpenfastFSI::predict_struct_states()
{
  timer_start(openFastTimer_);
  FAST.predict_states();
  timer_stop(openFastTimer_);
}

void
OpenfastFSI::predict_struct_timestep(const double curTime)
{
  send_loads(curTime);
  timer_start(openFastTimer_);
  FAST.update_states_driver_time_step();
  timer_stop(openFastTimer_);
}

void
OpenfastFSI::advance_struct_timestep(const double /* curTime */)
{

  timer_start(openFastTimer_);
  FAST.advance_to_next_driver_time_step();
  timer_stop(openFastTimer_);

  tStep_ += 1;

  // int nTurbinesGlob = FAST.get_nTurbinesGlob();
  // for (int i=0; i < nTurbinesGlob; i++) {
  //     if(fsiTurbineData_[i] != nullptr)
  //         fsiTurbineData_[i]->write_nc_def_loads(tStep_, curTime);
  // }
}

void
OpenfastFSI::send_loads(const double /* curTime */)
{

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) { // This may not be a turbine intended for
                                      // blade-resolved simulation
      int turbProc = fsiTurbineData_[i]->getProc();
      fsiTurbineData_[i]->mapLoads();

      int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
      if (bulk_->parallel_rank() == turbProc) {
        MPI_Reduce(
          MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.twr_ld.data(),
          (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, MPI_SUM,
          turbProc, bulk_->parallel());
        for (int k = 0; k < (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6; k++)
          fsiTurbineData_[i]->brFSIdata_.twr_ld[k] =
            fsiTurbineData_[i]->brFSIdata_.twr_ld[k];
        FAST.setTowerForces(fsiTurbineData_[i]->brFSIdata_.twr_ld, i);

        MPI_Reduce(
          MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.bld_ld.data(),
          nTotBldNodes * 6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_->parallel());
        for (int k = 0; k < nTotBldNodes * 6; k++)
          fsiTurbineData_[i]->brFSIdata_.bld_ld[k] =
            fsiTurbineData_[i]->brFSIdata_.bld_ld[k];

        FAST.setBladeForces(fsiTurbineData_[i]->brFSIdata_.bld_ld, i);

      } else {
        MPI_Reduce(
          fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), NULL,
          (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, MPI_SUM,
          turbProc, bulk_->parallel());
        MPI_Reduce(
          fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), NULL,
          (nTotBldNodes) * 6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_->parallel());
      }
    }
  }
}

void
OpenfastFSI::get_displacements(double /* current_time */)
{

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) { // This may not be a turbine intended for
                                      // blade-resolved simulation
      int turbProc = fsiTurbineData_[i]->getProc();
      if (bulk_->parallel_rank() == turbProc) {
        FAST.getTowerDisplacements(
          fsiTurbineData_[i]->brFSIdata_.twr_def.data(),
          fsiTurbineData_[i]->brFSIdata_.twr_vel.data(), i);
        FAST.getBladeDisplacements(
          fsiTurbineData_[i]->brFSIdata_.bld_def.data(),
          fsiTurbineData_[i]->brFSIdata_.bld_vel.data(), i);
        FAST.getBladeRootDisplacements(
          fsiTurbineData_[i]->brFSIdata_.bld_root_def.data(), i);
        FAST.getBladePitch(fsiTurbineData_[i]->brFSIdata_.bld_pitch.data(), i);
        FAST.getHubDisplacement(
          fsiTurbineData_[i]->brFSIdata_.hub_def.data(),
          fsiTurbineData_[i]->brFSIdata_.hub_vel.data(), i);
        FAST.getNacelleDisplacement(
          fsiTurbineData_[i]->brFSIdata_.nac_def.data(),
          fsiTurbineData_[i]->brFSIdata_.nac_vel.data(), i);
        
        std::ofstream nacelle_loc_file("nacelle_loc.dat", std::ios_base::out);
        auto nacelle_orient = vs::rotation_tensor(fsiTurbineData_[i]->brFSIdata_.nac_def);
        for (int k = 0; k < 3; k++)
          nacelle_loc_file << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[k] + fsiTurbineData_[i]->brFSIdata_.nac_def[k] << " " ;
        nacelle_loc_file << std::endl;
        nacelle_loc_file << nacelle_orient.xx() << " " << nacelle_orient.xy() << " " << nacelle_orient.xz() << " " << std::endl;
        nacelle_loc_file << nacelle_orient.yx() << " " << nacelle_orient.yy() << " " << nacelle_orient.yz() << " " << std::endl;
        nacelle_loc_file << nacelle_orient.zx() << " " << nacelle_orient.zy() << " " << nacelle_orient.zz() << " " << std::endl;
        nacelle_loc_file.close();
      }

      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.twr_def.data(),
        (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());
      int numBlades = fsiTurbineData_[i]->params_.numBlades;
      int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_def.data(), nTotBldNodes * 6,
        MPI_DOUBLE, turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_root_def.data(), numBlades * 6,
        MPI_DOUBLE, turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_pitch.data(), numBlades, MPI_DOUBLE,
        turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.hub_def.data(), 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.nac_def.data(), 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.twr_vel.data(),
        (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.bld_vel.data(), nTotBldNodes * 6,
        MPI_DOUBLE, turbProc, bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.hub_vel.data(), 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());
      MPI_Bcast(
        fsiTurbineData_[i]->brFSIdata_.nac_vel.data(), 6, MPI_DOUBLE, turbProc,
        bulk_->parallel());

      if (bulk_->parallel_rank() == turbProc) {
        std::ofstream bld_bm_mesh;
        bld_bm_mesh.open("blade_beam_mesh_naluwind.csv", std::ios_base::out);
        for (int k = 0; k < nTotBldNodes; k++) {
          bld_bm_mesh
            << fsiTurbineData_[i]->brFSIdata_.bld_ref_pos[k * 6] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_ref_pos[k * 6 + 1] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_ref_pos[k * 6 + 2] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_def[k * 6] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_def[k * 6 + 1] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_def[k * 6 + 2] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_def[k * 6 + 3] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_def[k * 6 + 4] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_def[k * 6 + 5] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_vel[k * 6] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_vel[k * 6 + 1] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_vel[k * 6 + 2] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_vel[k * 6 + 3] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_vel[k * 6 + 4] << ","
            << fsiTurbineData_[i]->brFSIdata_.bld_vel[k * 6 + 5] << std::endl;
        }

        bld_bm_mesh << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[0] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[1] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_ref_pos[2] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_def[0] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_def[1] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_def[2] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_def[3] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_def[4] << ","
                    << fsiTurbineData_[i]->brFSIdata_.nac_def[5] << std::endl;

        bld_bm_mesh.close();
      }

      // fsiTurbineData_[i]->setSampleDisplacement(current_time);

      // For testing purposes
      // fsiTurbineData_[i]->setRefDisplacement(current_time);
    }
  }
}

void
OpenfastFSI::compute_div_mesh_velocity()
{

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  timer_start(naluTimer_);
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) // This may not be a turbine intended for
                                    // blade-resolved simulation
      fsiTurbineData_[i]->compute_div_mesh_velocity();
  }
  timer_stop(naluTimer_);
}

void
OpenfastFSI::set_rotational_displacement(
  std::array<double, 3> axis, double omega, double curTime)
{

  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) // This may not be a turbine intended for
                                    // blade-resolved simulation
      fsiTurbineData_[i]->setRotationDisplacement(axis, omega, curTime);
  }
}
void
OpenfastFSI::map_displacements(double current_time, bool updateCurCoor)
{

  timer_start(naluTimer_);
  get_displacements(current_time);

  stk::mesh::Selector sel;
  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != NULL) {
      fsiTurbineData_[i]->mapDisplacements(current_time);
      sel &= stk::mesh::selectUnion(fsiTurbineData_[i]->getPartVec());
    }
  }

  if (updateCurCoor) {
    auto& meta = bulk_->mesh_meta_data();
    const VectorFieldType* modelCoords =
      meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
    VectorFieldType* curCoords =
      meta.get_field<double>(stk::topology::NODE_RANK, "current_coordinates");
    VectorFieldType* displacement =
      meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");

    modelCoords->sync_to_host();
    curCoords->sync_to_host();
    displacement->sync_to_host();

    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        for (size_t in = 0; in < b->size(); in++) {

          double* cc = stk::mesh::field_data(*curCoords, node);
          double* mc = stk::mesh::field_data(*modelCoords, node);
          double* cd = stk::mesh::field_data(*displacement, node);

          for (int j = 0; j < 3; ++j) {
            cc[j] = mc[j] + cd[j];
          }
        }
      }
    }

    curCoords->modify_on_host();
    curCoords->sync_to_device();
  }
  timer_stop(naluTimer_);
}

void
OpenfastFSI::map_loads(const int tStep, const double curTime)
{
  timer_start(naluTimer_);
  int nTurbinesGlob = FAST.get_nTurbinesGlob();
  for (int i = 0; i < nTurbinesGlob; i++) {
    if (fsiTurbineData_[i] != nullptr) { // This may not be a turbine intended
                                         // for blade-resolved simulation
      int turbProc = fsiTurbineData_[i]->getProc();
      fsiTurbineData_[i]->mapLoads();
      int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
      if (bulk_->parallel_rank() == turbProc) {
        MPI_Reduce(
          MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.twr_ld.data(),
          (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, MPI_SUM,
          turbProc, bulk_->parallel());
        MPI_Reduce(
          MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.bld_ld.data(),
          nTotBldNodes * 6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_->parallel());
      } else {
        MPI_Reduce(
          fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), NULL,
          (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, MPI_SUM,
          turbProc, bulk_->parallel());
        MPI_Reduce(
          fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), NULL,
          (nTotBldNodes) * 6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_->parallel());
      }

      fsiTurbineData_[i]->write_nc_def_loads(tStep, curTime);
    }
  }
  timer_stop(naluTimer_);
}

void
OpenfastFSI::timer_start(std::pair<double, double>& timer)
{
  timer.first = NaluEnv::self().nalu_time();
}

void
OpenfastFSI::timer_stop(std::pair<double, double>& timer)
{
  timer.first = NaluEnv::self().nalu_time() - timer.first;
  timer.second += timer.first;
}

} // namespace nalu

} // namespace sierra
