// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "wind_energy/LidarPatterns.h"

#include "NaluParsedTypes.h"
#include "NaluParsing.h"
#include "master_element/TensorOps.h"

#include "xfer/Transfer.h"
#include "xfer/LocalVolumeSearch.h"
#include "netcdf.h"
#include "Ioss_FileInfo.h"

#include <memory>

namespace sierra {
namespace nalu {

constexpr int dim = 3;

SegmentType
segment_generator_types(std::string name)
{
  std::transform(name.cbegin(), name.cend(), name.begin(), ::tolower);
  return std::map<std::string, SegmentType>{
    {"spinner", SegmentType::SPINNER},
    {"scanning", SegmentType::SCANNING},
  }
    .at(name);
}

std::unique_ptr<SegmentGenerator>
make_segment_generator(SegmentType type)
{
  switch (type) {
  case SegmentType::SPINNER:
    return std::make_unique<SpinnerLidarSegmentGenerator>();
  case SegmentType::SCANNING:
    return std::make_unique<ScanningLidarSegmentGenerator>();
  default:
    throw std::runtime_error("Invalid lidar type");
    return std::make_unique<SpinnerLidarSegmentGenerator>();
  }
}

std::unique_ptr<SegmentGenerator>
make_segment_generator(const std::string& name)
{
  return make_segment_generator(segment_generator_types(name));
}

namespace {

std::array<double, 3>
to_array3(Coordinates x)
{
  return {x.x_, x.y_, x.z_};
}

std::array<double, 3>
rotate_euler_vec(
  const std::array<double, 3>& axis, double angle, std::array<double, 3> vec)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  normalize_vec3(vec.data());
  std::array<double, 9> nX = {
    {0, -axis[ZH], +axis[YH], +axis[ZH], 0, -axis[XH], -axis[YH], +axis[XH],
     0}};
  const double cosTheta = std::cos(angle);
  std::array<double, 9> rot = {
    {cosTheta, 0, 0, 0, cosTheta, 0, 0, 0, cosTheta}};

  const double sinTheta = std::sin(angle);
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      rot[j * 3 + i] +=
        (1 - cosTheta) * axis[i] * axis[j] + sinTheta * nX[j * 3 + i];
    }
  }
  std::array<double, 3> vecprime;
  matvec33(rot.data(), vec.data(), vecprime.data());
  return vecprime;
}

std::array<double, 3>
reflect(const std::array<double, 3>& line, const std::array<double, 3>& vec)
{
  enum { XH = 0, YH = 1, ZH = 2 };

  std::array<double, 9> ref = {
    {1 - 2 * line[XH] * line[XH], -2 * line[XH] * line[YH],
     -2 * line[XH] * line[ZH], -2 * line[YH] * line[XH],
     1 - 2 * line[YH] * line[YH], -2 * line[YH] * line[ZH],
     -2 * line[ZH] * line[XH], -2 * line[ZH] * line[YH],
     1 - 2 * line[ZH] * line[ZH]}};

  std::array<double, 3> result;
  matvec33(ref.data(), vec.data(), result.data());
  return result;
}

auto
affine(
  const std::array<double, 3>& trans,
  const std::array<double, 9>& lin,
  const std::array<double, 3>& x)
{
  enum {
    XX = 0,
    XY = 1,
    XZ = 2,
    YX = 3,
    YY = 4,
    YZ = 5,
    ZX = 6,
    ZY = 7,
    ZZ = 8
  };
  enum { XH = 0, YH = 1, ZH = 2 };
  return std::array<double, 3>{
    trans[XH] + (lin[XX] * x[XH] + lin[XY] * x[YH] + lin[XZ] * x[ZH]),
    trans[YH] + (lin[YX] * x[XH] + lin[YY] * x[YH] + lin[YZ] * x[ZH]),
    trans[ZH] + (lin[ZX] * x[XH] + lin[ZY] * x[YH] + lin[ZZ] * x[ZH])};
}

auto
affine(
  const std::array<double, 3>& trans,
  double lin,
  const std::array<double, 3>& x)
{
  return affine(
    trans,
    {
      lin,
      0,
      0,
      0,
      lin,
      0,
      0,
      0,
      lin,
    },
    x);
}

} // namespace

void
ScanningLidarSegmentGenerator::load(const YAML::Node& node)
{
  center_ = to_array3(node["center"].as<Coordinates>());

  axis_ = to_array3(node["axis"].as<Coordinates>());
  normalize_vec3(axis_.data());

  double sweep_angle_in_degrees = 20;
  get_if_present(node, "sweep_angle", sweep_angle_in_degrees);
  ThrowRequireMsg(sweep_angle_in_degrees > 0, "Sweep angle must be positive");
  sweep_angle_ = convert::degrees_to_radians(sweep_angle_in_degrees);

  double step_delta_angle_in_degrees = 1;
  get_if_present(node, "step_delta_angle", step_delta_angle_in_degrees);
  step_delta_angle_ = convert::degrees_to_radians(step_delta_angle_in_degrees);

  ThrowRequireMsg(step_delta_angle_ > 0, "step delta angle must be positive");
  ThrowRequireMsg(
    step_delta_angle_ <= sweep_angle_,
    "step delta angle must be less than full sweep");

  get_if_present(node, "stare_time", stare_time_);
  ThrowRequireMsg(stare_time_ > 0, "stare time must be positive");

  get_if_present(node, "reset_time_delta", reset_time_delta_);
  ThrowRequireMsg(
    reset_time_delta_ >= 0, "reset time delta must be semi-positive");

  get_required(node, "beam_length", beam_length_);

  if (node["ground_direction"]) {
    ground_normal_ = to_array3(node["ground_direction"].as<Coordinates>());
    normalize_vec3(ground_normal_.data());
  }

  if (node["elevation_angles"]) {
    elevation_table_ = node["elevation_angles"].as<std::vector<double>>();
    std::transform(
      elevation_table_.cbegin(), elevation_table_.cend(),
      elevation_table_.begin(), convert::degrees_to_radians);
  }

  end_of_forward_phase_ = determine_end_of_forward_phase();
}

double
ScanningLidarSegmentGenerator::periodic_time(double time) const
{
  const double total_sweep_time = end_of_forward_phase_ + reset_time_delta_;
  return time - std::floor(time / total_sweep_time) * total_sweep_time;
}

int
ScanningLidarSegmentGenerator::periodic_count(double time) const
{
  const double total_sweep_time = end_of_forward_phase_ + reset_time_delta_;
  return std::floor(time / total_sweep_time);
}

double
ScanningLidarSegmentGenerator::determine_elevation_angle(int count) const
{
  const int orientation = (dir_ == direction::CLOCKWISE) ? -1 : 1;
  return orientation * elevation_table_.at(count % elevation_table_.size());
}

ScanningLidarSegmentGenerator::phase
ScanningLidarSegmentGenerator::determine_operation_phase(
  double periodic_time) const
{
  return (periodic_time <= end_of_forward_phase_) ? phase::FORWARD
                                                  : phase::RESET;
}

double
ScanningLidarSegmentGenerator::angle_if_during_reset(double periodic_time) const
{
  const double reset_time = periodic_time - end_of_forward_phase_;
  const int orientation = (dir_ == direction::CLOCKWISE) ? -1 : 1;
  return orientation *
         (sweep_angle_ / 2 - (sweep_angle_ / reset_time_delta_) * reset_time);
}

double
ScanningLidarSegmentGenerator::angle_if_during_forward_phase(
  double periodic_time) const
{
  const int orientation = (dir_ == direction::CLOCKWISE) ? -1 : 1;
  return orientation *
         (-sweep_angle_ / 2 +
          std::floor(periodic_time / stare_time_) * step_delta_angle_);
}

double
ScanningLidarSegmentGenerator::determine_current_angle(
  double periodic_time) const
{
  return (determine_operation_phase(periodic_time) == phase::FORWARD)
           ? angle_if_during_forward_phase(periodic_time)
           : angle_if_during_reset(periodic_time);
}

std::array<double, 3>
cross(std::array<double, 3> u, std::array<double, 3> v)
{
  return {
    u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]};
}

Segment
ScanningLidarSegmentGenerator::generate(double time) const
{
  /*
   scanning lidar steps to a particular angle in a sweep, stares, then moves
   quickly to the next angle. At the max sweep angle, it resets itself to the
   starting angle (-sweep angle/2) over some finite period of time.
  */
  const auto tail = center_;
  const auto yaw_angle = determine_current_angle(periodic_time(time));
  const auto pitch_angle = determine_elevation_angle(periodic_count(time));

  const auto yaxis = cross(axis_, ground_normal_);
  const auto xprime = rotate_euler_vec(yaxis, pitch_angle, axis_);
  const auto zprime = rotate_euler_vec(yaxis, pitch_angle, ground_normal_);
  const auto sight_vector = rotate_euler_vec(zprime, yaw_angle, xprime);

  const auto tip = affine(center_, beam_length_, sight_vector);
  return Segment{tip, tail};
}

void
SpinnerLidarSegmentGenerator::load(const YAML::Node& node)
{
  NaluEnv::self().naluOutputP0()
    << "LidarLineOfSite::SpinnerLidarSegmentGenerator::load" << std::endl;

  ThrowRequireMsg(node["center"], "Lidar center must be provided");
  ThrowRequireMsg(node["axis"], "Lidar axis must be provided");

  lidarCenter_ = to_array3(node["center"].as<Coordinates>());

  laserAxis_ = to_array3(node["axis"].as<Coordinates>());
  normalize_vec3(laserAxis_.data());

  double innerPrismTheta0 = 90;
  get_if_present(node, "inner_prism_initial_theta", innerPrismTheta0);

  double innerPrismRot = 3.5;
  get_if_present(node, "inner_prism_rotation_rate", innerPrismRot);

  double innerPrismAzi = 15.2;
  get_if_present(node, "inner_prism_azimuth", innerPrismAzi);

  double outerPrismTheta0 = 90;
  get_if_present(node, "outer_prism_initial_theta", outerPrismTheta0);

  double outerPrismRot = 6.5;
  get_if_present(node, "outer_prism_rotation_rate", outerPrismRot);

  double outerPrismAzi = 15.2;
  get_if_present(node, "outer_prism_azimuth", outerPrismAzi);

  innerPrism_ = {
    convert::degrees_to_radians(innerPrismTheta0),
    convert::rotations_to_radians(innerPrismRot),
    convert::degrees_to_radians(innerPrismAzi)};
  outerPrism_ = {
    convert::degrees_to_radians(outerPrismTheta0),
    convert::rotations_to_radians(outerPrismRot),
    convert::degrees_to_radians(outerPrismAzi)};
  get_required(node, "beam_length", beamLength_);

  if (node["ground_direction"]) {
    groundNormal_ = to_array3(node["ground_direction"].as<Coordinates>());
    normalize_vec3(groundNormal_.data());
  }

  ThrowRequireMsg(
    std::abs(ddot(groundNormal_.data(), laserAxis_.data(), 3)) <
      small_positive_value(),
    "Ground and laser axes must be orthogonal");
}

Segment
SpinnerLidarSegmentGenerator::generate(double time) const
{
  const auto inner_angle = -(innerPrism_.azimuth_ / 2 + M_PI_2);
  const auto outer_angle = outerPrism_.azimuth_ / 2;

  const auto reflection_1 = rotate_euler_vec(
    laserAxis_, innerPrism_.theta(time),
    rotate_euler_vec(groundNormal_, inner_angle, laserAxis_));

  const auto reflection_2 = rotate_euler_vec(
    laserAxis_, outerPrism_.theta(time),
    rotate_euler_vec(groundNormal_, outer_angle, laserAxis_));

  const auto tail = lidarCenter_;
  const auto tip = affine(
    tail, -beamLength_,
    reflect(reflection_2, reflect(reflection_1, laserAxis_)));
  return Segment{tip, tail};
}

} // namespace nalu
} // namespace sierra
