// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/actuator/ActuatorBulk.h>
#include <aero/actuator/ActuatorBulkSimple.h>
#include <NaluParsing.h>
#include <aero/actuator/ActuatorParsingSimple.h>
#include <aero/actuator/ActuatorParsing.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

namespace {

/** Resizes a vector to size N
 *
 * - If vec is of size 1, resizes a vector to size N (assuming all
 *   constant values).
 * - If vec is of size N, do nothing.
 *
 * Similar to std::vector::resize(), except this returns helpful
 * error message if vec is not size 1 or N.  Useful when validating
 * user input.
 *
 */
std::vector<double>
extend_double_vector(std::vector<double> vec, const unsigned N)
{
  if ((vec.size() != 1) && (vec.size() != N))
    throw std::runtime_error("Vector is not of size 1 or " + std::to_string(N));
  if (vec.size() == 1) { // Extend the vector to size N
    std::vector<double> newvec(N, vec[0]);
    return newvec;
  }
  if (vec.size() == N)
    return vec;
  return vec; // Should not get here
}
} // namespace

ActuatorMetaSimple
actuator_Simple_parse(const YAML::Node& y_node, const ActuatorMeta& actMeta)
{
  ActuatorMetaSimple actMetaSimple(actMeta);

  actMetaSimple.dR_.modify_host();

  NaluEnv::self().naluOutputP0()
    << "In actuator_Simple_parse() " << std::endl; // LCCOUT

  const YAML::Node y_actuator = y_node["actuator"];
  ThrowErrorMsgIf(
    !y_actuator, "actuator argument is "
                 "missing from yaml node passed to actuator_Simple_parse");

  // Load the debug option
  const YAML::Node debug_output = y_actuator["debug_output"];
  if (debug_output)
    actMetaSimple.debug_output_ = debug_output.as<bool>();
  else
    actMetaSimple.debug_output_ = false;

  // Load the spread force option option
  const YAML::Node useSpreadActF = y_actuator["useSpreadActuatorForce"];
  if (useSpreadActF)
    actMetaSimple.useSpreadActuatorForce = useSpreadActF.as<bool>();
  else
    actMetaSimple.useSpreadActuatorForce = true;

  size_t n_simpleblades_;
  get_required(y_actuator, "n_simpleblades", n_simpleblades_);
  actMetaSimple.n_simpleblades_ = n_simpleblades_;

  // Declare some vectors to store blade information
  std::vector<std::vector<double>> input_chord_table;
  std::vector<std::vector<double>> input_twist_table;
  std::vector<std::vector<double>> input_elem_area;
  std::vector<std::vector<double>> input_aoa_polartable;
  std::vector<std::vector<double>> input_cl_polartable;
  std::vector<std::vector<double>> input_cd_polartable;

  if (actMetaSimple.n_simpleblades_ > 0) {
    actMetaSimple.numPointsTotal_ = 0;
    for (unsigned iBlade = 0; iBlade < n_simpleblades_; iBlade++) {

      const YAML::Node cur_blade = y_actuator["Blade" + std::to_string(iBlade)];
      get_if_present_no_default(
        cur_blade, "output_file_name", actMetaSimple.output_filenames_[iBlade]);
      if (
        !actMetaSimple.output_filenames_[iBlade].empty() &&
        NaluEnv::self().parallel_rank() == (int)iBlade) {
        actMetaSimple.has_output_file_ = true;
      }

      size_t num_force_pts_blade;
      get_required(cur_blade, "num_force_pts_blade", num_force_pts_blade);
      actMetaSimple.num_force_pts_blade_.h_view(iBlade) = num_force_pts_blade;
      actMetaSimple.numPointsTurbine_.h_view(iBlade) = num_force_pts_blade;
      actMetaSimple.numPointsTotal_ += num_force_pts_blade;

      if (num_force_pts_blade > actMetaSimple.max_num_force_pts_blade_) {
        actMetaSimple.max_num_force_pts_blade_ = num_force_pts_blade;
      }

      if (actMetaSimple.debug_output_)
        NaluEnv::self().naluOutputP0()
          << "Reading blade: " << iBlade << " num_force_pts_blade: "
          << actMetaSimple.numPointsTurbine_.h_view(iBlade)
          << std::endl; // LCCOUT

      epsilon_parsing(iBlade, cur_blade, actMetaSimple);

      // Handle blade properties
      // p1
      std::vector<double> p1Temp(3);
      get_required(cur_blade, "p1", p1Temp);
      for (int j = 0; j < 3; j++) {
        actMetaSimple.p1_.h_view(iBlade, j) = p1Temp[j];
      }
      // p2
      std::vector<double> p2Temp(3);
      get_required(cur_blade, "p2", p2Temp);
      for (int j = 0; j < 3; j++) {
        actMetaSimple.p2_.h_view(iBlade, j) = p2Temp[j];
      }
      // p1zeroAOA
      std::vector<double> p1zeroAOATemp(3);
      Coordinates p1zeroAOA;
      get_required(cur_blade, "p1_zero_alpha_dir", p1zeroAOATemp);
      double p1zeroAOAnorm = sqrt(
        p1zeroAOATemp[0] * p1zeroAOATemp[0] +
        p1zeroAOATemp[1] * p1zeroAOATemp[1] +
        p1zeroAOATemp[2] * p1zeroAOATemp[2]);
      for (int j = 0; j < 3; j++) {
        actMetaSimple.p1ZeroAlphaDir_.h_view(iBlade, j) =
          p1zeroAOATemp[j] / p1zeroAOAnorm;
      }
      p1zeroAOA.x_ = actMetaSimple.p1ZeroAlphaDir_.h_view(iBlade, 0);
      p1zeroAOA.y_ = actMetaSimple.p1ZeroAlphaDir_.h_view(iBlade, 1);
      p1zeroAOA.z_ = actMetaSimple.p1ZeroAlphaDir_.h_view(iBlade, 2);

      // Calculate some stuff
      // Span direction
      Coordinates spanDir;
      spanDir.x_ = p2Temp[0] - p1Temp[0];
      spanDir.y_ = p2Temp[1] - p1Temp[1];
      spanDir.z_ = p2Temp[2] - p1Temp[2];
      double spandirnorm = sqrt(
        spanDir.x_ * spanDir.x_ + spanDir.y_ * spanDir.y_ +
        spanDir.z_ * spanDir.z_);
      spanDir.x_ = spanDir.x_ / spandirnorm;
      spanDir.y_ = spanDir.y_ / spandirnorm;
      spanDir.z_ = spanDir.z_ / spandirnorm;
      actMetaSimple.spanDir_.h_view(iBlade, 0) = spanDir.x_;
      actMetaSimple.spanDir_.h_view(iBlade, 1) = spanDir.y_;
      actMetaSimple.spanDir_.h_view(iBlade, 2) = spanDir.z_;
      // Chord normal direction
      Coordinates chodrNormalDir;
      chodrNormalDir.x_ = p1zeroAOA.y_ * spanDir.z_ - p1zeroAOA.z_ * spanDir.y_;
      chodrNormalDir.y_ = p1zeroAOA.z_ * spanDir.x_ - p1zeroAOA.x_ * spanDir.z_;
      chodrNormalDir.z_ = p1zeroAOA.x_ * spanDir.y_ - p1zeroAOA.y_ * spanDir.x_;
      actMetaSimple.chordNormalDir_.h_view(iBlade, 0) = chodrNormalDir.x_;
      actMetaSimple.chordNormalDir_.h_view(iBlade, 1) = chodrNormalDir.y_;
      actMetaSimple.chordNormalDir_.h_view(iBlade, 2) = chodrNormalDir.z_;

      // output directions
      if (actMetaSimple.debug_output_) {
        NaluEnv::self().naluOutputP0() // LCCOUT
          << "Blade: " << iBlade << " p1zeroAOA dir: " << p1zeroAOA.x_ << " "
          << p1zeroAOA.y_ << " " << p1zeroAOA.z_ << std::endl;
        NaluEnv::self().naluOutputP0() // LCCOU
          << "Blade: " << iBlade << " Span dir: " << spanDir.x_ << " "
          << spanDir.y_ << " " << spanDir.z_ << std::endl;
        NaluEnv::self().naluOutputP0() // LCCOUT
          << "Blade: " << iBlade << " chord norm dir: " << std::setprecision(5)
          << chodrNormalDir.x_ << " " << chodrNormalDir.y_ << " "
          << chodrNormalDir.z_ << std::endl;
      }

      // Chord definitions
      const YAML::Node chord_table = cur_blade["chord_table"];
      std::vector<double> chordtemp;
      if (chord_table)
        chordtemp = chord_table.as<std::vector<double>>();
      else
        throw std::runtime_error("ActuatorSimpleNGP: missing chord_table");
      std::vector<double> chord_table_extended =
        extend_double_vector(chordtemp, num_force_pts_blade);
      input_chord_table.push_back(chord_table_extended);

      // twist definitions
      const YAML::Node twist_table = cur_blade["twist_table"];
      std::vector<double> twisttemp;
      if (twist_table)
        twisttemp = twist_table.as<std::vector<double>>();
      else
        throw std::runtime_error("ActuatorSimpleNGP: missing twist_table");
      input_twist_table.push_back(
        extend_double_vector(twisttemp, num_force_pts_blade));

      // Calculate elem areas
      std::vector<double> elemareatemp(num_force_pts_blade, 0.0);
      std::vector<double> dx(3, 0.0);
      for (int i = 0; i < 3; i++)
        dx[i] = (p2Temp[i] - p1Temp[i]) / (double)num_force_pts_blade;
      double dx_norm = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
      actMetaSimple.dR_.h_view(iBlade) = dx_norm;
      for (unsigned i = 0; i < num_force_pts_blade; i++)
        elemareatemp[i] = dx_norm * chord_table_extended[i];
      input_elem_area.push_back(elemareatemp);

      // Polar tables
      // --- aoa ---
      const YAML::Node aoa_table = cur_blade["aoa_table"];
      std::vector<double> aoatemp;
      if (aoa_table)
        aoatemp = aoa_table.as<std::vector<double>>();
      else
        throw std::runtime_error("ActuatorSimpleNGP: missing aoa_table");
      input_aoa_polartable.push_back(aoatemp);
      size_t polartableN = aoatemp.size();
      actMetaSimple.polarTableSize_.h_view(iBlade) = polartableN;
      // get the maximum size
      if (polartableN > actMetaSimple.maxPolarTableSize_) {
        actMetaSimple.maxPolarTableSize_ = polartableN;
      }

      // --- cl ---
      const YAML::Node cl_table = cur_blade["cl_table"];
      std::vector<double> cltemp;
      if (cl_table)
        cltemp = cl_table.as<std::vector<double>>();
      else
        throw std::runtime_error("ActuatorSimpleNGP: missing cl_table");
      input_cl_polartable.push_back(extend_double_vector(cltemp, polartableN));

      // --- cd ---
      const YAML::Node cd_table = cur_blade["cd_table"];
      std::vector<double> cdtemp;
      if (cd_table)
        cdtemp = cd_table.as<std::vector<double>>();
      else
        throw std::runtime_error("ActuatorSimpleNGP: missing cd_table");
      input_cd_polartable.push_back(extend_double_vector(cdtemp, polartableN));

    } // End loop over blades
  } else {
    throw std::runtime_error("Number of simple blades <= 0 ");
  }

  // resize blade definition tables
  Act2DArrayDblDv chordview(
    "chord_table_view", n_simpleblades_,
    actMetaSimple.max_num_force_pts_blade_);
  Act2DArrayDblDv twistview(
    "twist_table_view", n_simpleblades_,
    actMetaSimple.max_num_force_pts_blade_);
  Act2DArrayDblDv elem_area_view(
    "elem_area_view", n_simpleblades_, actMetaSimple.max_num_force_pts_blade_);
  actMetaSimple.chord_tableDv_ = chordview;
  actMetaSimple.twistTableDv_ = twistview;
  actMetaSimple.elemAreaDv_ = elem_area_view;
  // Copy the information over
  for (unsigned iBlade = 0; iBlade < n_simpleblades_; iBlade++) {
    for (int j = 0; j < actMetaSimple.numPointsTurbine_.h_view(iBlade); j++) {
      actMetaSimple.chord_tableDv_.h_view(iBlade, j) =
        input_chord_table[iBlade][j];
      actMetaSimple.twistTableDv_.h_view(iBlade, j) =
        input_twist_table[iBlade][j];
      actMetaSimple.elemAreaDv_.h_view(iBlade, j) = input_elem_area[iBlade][j];
    }
  }

  // resize the polar table views
  Act2DArrayDblDv aoaview(
    "aoa_polartable_view", n_simpleblades_, actMetaSimple.maxPolarTableSize_);
  Act2DArrayDblDv clview(
    "cl_polartable_view", n_simpleblades_, actMetaSimple.maxPolarTableSize_);
  Act2DArrayDblDv cdview(
    "cd_polartable_view", n_simpleblades_, actMetaSimple.maxPolarTableSize_);
  actMetaSimple.aoaPolarTableDv_ = aoaview;
  actMetaSimple.clPolarTableDv_ = clview;
  actMetaSimple.cdPolarTableDv_ = cdview;
  // Copy the information over
  for (unsigned iBlade = 0; iBlade < n_simpleblades_; iBlade++) {
    for (int j = 0; j < actMetaSimple.polarTableSize_.h_view(iBlade); j++) {
      actMetaSimple.aoaPolarTableDv_.h_view(iBlade, j) =
        input_aoa_polartable[iBlade][j];
      actMetaSimple.clPolarTableDv_.h_view(iBlade, j) =
        input_cl_polartable[iBlade][j];
      actMetaSimple.cdPolarTableDv_.h_view(iBlade, j) =
        input_cd_polartable[iBlade][j];
    }
  }
  if (actMetaSimple.debug_output_) {
    NaluEnv::self().naluOutputP0()
      << " actMetaSimple.numPointsTotal_ = " << actMetaSimple.numPointsTotal_
      << std::endl; // LCCOUT
    NaluEnv::self().naluOutputP0()
      << "Done actuator_Simple_parse()" << std::endl; // LCCOUT
  }
  return actMetaSimple;
}

} // namespace nalu
} // namespace sierra
