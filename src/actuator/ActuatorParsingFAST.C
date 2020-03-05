// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <actuator/ActuatorBulk.h>
#include <actuator/ActuatorBulkFAST.h>
#include <NaluParsing.h>
#include <actuator/ActuatorParsingFAST.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

namespace {
void
readTurbineData(int iTurb, ActuatorMetaFAST& actMetaFAST, YAML::Node turbNode)
{
  fast::fastInputs& fi = actMetaFAST.fastInputs_;
  // Read turbine data for a given turbine using the YAML node
  get_required(turbNode, "turb_id", fi.globTurbineData[iTurb].TurbID);
  get_required(
    turbNode, "fast_input_filename",
    fi.globTurbineData[iTurb].FASTInputFileName);
  get_required(
    turbNode, "restart_filename",
    fi.globTurbineData[iTurb].FASTRestartFileName);

  get_required(
    turbNode, "turbine_base_pos", fi.globTurbineData[iTurb].TurbineBasePos);
  get_required(
    turbNode, "turbine_hub_pos", fi.globTurbineData[iTurb].TurbineHubPos);
  get_required(
    turbNode, "num_force_pts_blade",
    fi.globTurbineData[iTurb].numForcePtsBlade);
  get_required(
    turbNode, "num_force_pts_tower", fi.globTurbineData[iTurb].numForcePtsTwr);

  get_if_present_no_default(turbNode, "nacelle_cd", fi.globTurbineData[iTurb].nacelle_cd);
  get_if_present_no_default(
    turbNode, "nacelle_area", fi.globTurbineData[iTurb].nacelle_area);
  get_if_present_no_default(
    turbNode, "air_density", fi.globTurbineData[iTurb].air_density);

  int numBlades=3;
  get_if_present_no_default(turbNode, "num_blades", numBlades);
  ThrowErrorMsgIf(numBlades!=3,"ERROR::ActuatorParsingFAST::Currently only 3 bladed turbines are supported.");

  actMetaFAST.numPointsTurbine_.h_view(iTurb) =
    1 // hub
    + fi.globTurbineData[iTurb].numForcePtsTwr +
    fi.globTurbineData[iTurb].numForcePtsBlade *
      numBlades;
  actMetaFAST.numPointsTotal_+=actMetaFAST.numPointsTurbine_.h_view(iTurb);
}
} // namespace

ActuatorMetaFAST
actuator_FAST_parse(const YAML::Node& y_node, const ActuatorMeta& actMeta, double naluTimeStep)
{
  ActuatorMetaFAST actMetaFAST(actMeta);
  fast::fastInputs& fi = actMetaFAST.fastInputs_;
  fi.comm = NaluEnv::self().parallel_comm();
  fi.nTurbinesGlob = actMetaFAST.numberOfActuators_;

  const YAML::Node y_actuator = y_node["actuator"];
  ThrowErrorMsgIf(
    !y_actuator, "actuator argument is "
                 "missing from yaml node passed to actuator_FAST_parse");
  if (fi.nTurbinesGlob > 0) {
    get_if_present(y_actuator, "dry_run", fi.dryRun, false);
    get_if_present(y_actuator, "debug", fi.debug, false);
    get_required(y_actuator, "t_start", fi.tStart);
    std::string simStartType = "na";
    get_required(y_actuator, "simStart", simStartType);
    if (simStartType == "init") {
      if (fi.tStart == 0) {
        fi.simStart = fast::init;
      } else {
        throw std::runtime_error(
          "actuators: simStart type not consistent with start time for FAST");
      }
    } else if (simStartType == "trueRestart") {
      fi.simStart = fast::trueRestart;
    } else if (simStartType == "restartDriverInitFAST") {
      fi.simStart = fast::restartDriverInitFAST;
    }
    get_required(y_actuator, "n_every_checkpoint", fi.nEveryCheckPoint);
    get_required(y_actuator, "dt_fast", fi.dtFAST);

    actMetaFAST.timeStepRatio_ = naluTimeStep / fi.dtFAST;
    if (std::abs(naluTimeStep - actMetaFAST.timeStepRatio_ * fi.dtFAST) < 0.001) { // TODO: Fix
      // arbitrary number
      // 0.001
      NaluEnv::self().naluOutputP0()
          << "Time step ratio  dtNalu/dtFAST: " << actMetaFAST.timeStepRatio_ << std::endl;
    } else {
      throw std::runtime_error("ActuatorFAST: Ratio of Nalu's time step is not "
                               "an integral multiple of FAST time step");
    }

    get_required(y_actuator, "t_max", fi.tMax);

    if (y_actuator["super_controller"]) {
      get_required(y_actuator, "super_controller", fi.scStatus);
      get_required(y_actuator, "sc_libFile", fi.scLibFile);
      get_required(y_actuator, "num_sc_inputs", fi.numScInputs);
      get_required(y_actuator, "num_sc_outputs", fi.numScOutputs);
    }

    fi.globTurbineData.resize(fi.nTurbinesGlob);

    for (int iTurb = 0; iTurb < fi.nTurbinesGlob; iTurb++) {
      if (y_actuator["Turbine" + std::to_string(iTurb)]) {

        const YAML::Node cur_turbine =
          y_actuator["Turbine" + std::to_string(iTurb)];

        get_required(
          cur_turbine, "turbine_name", actMetaFAST.turbineNames_[iTurb]);

        std::string turbFileName;
        get_if_present(
          cur_turbine, "file_to_dump_turb_pts",
          actMetaFAST.turbineOutputFileNames_[iTurb]);

        get_if_present_no_default(
          cur_turbine, "fllt_correction",
          actMetaFAST.filterLiftLineCorrection_);

        // The value epsilon / chord [non-dimensional]
        // This is a vector containing the values for:
        //   - chord aligned (x),
        //   - tangential to chord (y),
        //   - spanwise (z)
        const YAML::Node epsilon_chord = cur_turbine["epsilon_chord"];
        const YAML::Node epsilon = cur_turbine["epsilon"];
        if (epsilon && epsilon_chord) {
          throw std::runtime_error(
            "epsilon and epsilon_chord have both been specified for Turbine " +
            std::to_string(iTurb) + "\nYou must pick one or the other.");
        }
        if (epsilon && actMetaFAST.filterLiftLineCorrection_) {
          throw std::runtime_error(
            "epsilon and fllt_correction have both been specified for "
            "Turbine " +
            std::to_string(iTurb) +
            "\nepsilon_chord and epsilon_min should be used with "
            "fllt_correction.");
        }

        // If epsilon/chord is given, store it,
        // If it is not given, set it to zero, such
        // that it is smaller than the standard epsilon and
        // will not be used
        std::vector<double> epsilonTemp(3);
        if (epsilon_chord) {
          // epsilon / chord
          epsilonTemp = epsilon_chord.as<std::vector<double>>();
          for (int j = 0; j < 3; j++) {
            actMetaFAST.epsilonChord_.h_view(iTurb, j) = epsilonTemp[j];
          }

          // Minimum epsilon allowed in simulation. This is required when
          //   specifying epsilon/chord
          get_required(cur_turbine, "epsilon_min", epsilonTemp);
          for (int j = 0; j < 3; j++) {
            actMetaFAST.epsilon_.h_view(iTurb, j) = epsilonTemp[j];
          }
        }
        // Set all unused epsilon values to zero
        else if (epsilon) {
          epsilonTemp = epsilon.as<std::vector<double>>();
          for (int j = 0; j < 3; j++) {
            actMetaFAST.epsilon_.h_view(iTurb, j) = epsilonTemp[j];
          }
        } else {
          throw std::runtime_error("ActuatorFAST: lacking epsilon vector");
        }

        // An epsilon value used for the tower
        const YAML::Node epsilon_tower = cur_turbine["epsilon_tower"];
        // If epsilon tower is given store it.
        // If not, use the standard epsilon value
        if (epsilon_tower) {
          epsilonTemp = epsilon_tower.as<std::vector<double>>();
          for (int j = 0; j < 3; j++) {
            actMetaFAST.epsilonTower_.h_view(iTurb, j) = epsilonTemp[j];
          }
        } else {
          for (int j = 0; j < 3; j++) {
            actMetaFAST.epsilonTower_.h_view(iTurb, j) =
              actMetaFAST.epsilon_.h_view(iTurb, j);
          }
        }

        readTurbineData(iTurb, actMetaFAST, cur_turbine);
      } else {
        throw std::runtime_error(
          "Node for Turbine" + std::to_string(iTurb) +
          " not present in input file or I cannot read it");
      }
    }

  } else {
    throw std::runtime_error("Number of turbines <= 0 ");
  }
  return actMetaFAST;
}

} // namespace nalu
} // namespace sierra
