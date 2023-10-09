// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "wind_energy/LidarPatterns.h"
#include "vs/vector.h"

#include "NaluParsedTypes.h"
#include "NaluParsing.h"
#include "master_element/TensorOps.h"

#include "xfer/Transfer.h"
#include "xfer/LocalVolumeSearch.h"
#include "netcdf.h"

#include <memory>

namespace sierra::nalu {

SegmentType
segment_generator_types(std::string name)
{
  std::transform(name.cbegin(), name.cend(), name.begin(), ::tolower);
  return std::map<std::string, SegmentType>{{"spinner", SegmentType::SPINNER},
                                            {"scanning", SegmentType::SCANNING},
                                            {"radar", SegmentType::RADAR}}
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
  case SegmentType::RADAR:
    return std::make_unique<RadarSegmentGenerator>();
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

vs::Vector
to_vec3(std::array<double, 3> x)
{
  return {x[0], x[1], x[2]};
}

vs::Vector
to_vec3(Coordinates x)
{
  return {x.x_, x.y_, x.z_};
}

std::array<double, 3>
rotate_euler_vec(
  const std::array<double, 3>& axis, double angle, std::array<double, 3> vec)
{
  enum { XH = 0, YH = 1, ZH = 2 };
  normalize_vec3(vec.data());
  std::array<double, 9> nX = {{0, -axis[ZH], +axis[YH], +axis[ZH], 0, -axis[XH],
                               -axis[YH], +axis[XH], 0}};
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
  get_if_present(
    node, "sweep_angle", sweep_angle_in_degrees, sweep_angle_in_degrees);
  ThrowRequireMsg(sweep_angle_in_degrees > 0, "Sweep angle must be positive");
  sweep_angle_ = convert::degrees_to_radians(sweep_angle_in_degrees);

  double step_delta_angle_in_degrees = 1;
  get_if_present(
    node, "step_delta_angle", step_delta_angle_in_degrees,
    step_delta_angle_in_degrees);
  step_delta_angle_ = convert::degrees_to_radians(step_delta_angle_in_degrees);

  ThrowRequireMsg(step_delta_angle_ > 0, "step delta angle must be positive");
  ThrowRequireMsg(
    step_delta_angle_ <= sweep_angle_,
    "step delta angle must be less than full sweep");

  get_if_present(node, "stare_time", stare_time_, stare_time_);
  ThrowRequireMsg(stare_time_ > 0, "stare time must be positive");

  get_if_present(
    node, "reset_time_delta", reset_time_delta_, reset_time_delta_);
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
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}

std::array<double, 3>
plane_sweep(
  const std::array<double, 3>& sweep_normal,
  const std::array<double, 3>& xprime,
  double yaw,
  double pitch)
{
  // normalize frequently for floating point, but these should all be unitary
  // operations
  auto plane_sight_vector = rotate_euler_vec(sweep_normal, yaw, xprime);
  normalize_vec3(plane_sight_vector.data());
  const auto yprime = cross(sweep_normal, plane_sight_vector);
  normalize_vec3(plane_sight_vector.data());
  auto sight = rotate_euler_vec(yprime, pitch, plane_sight_vector);
  normalize_vec3(sight.data());
  return sight;
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
  const auto yaw = determine_current_angle(periodic_time(time));
  const auto pitch = determine_elevation_angle(periodic_count(time));
  const auto sight_vector = plane_sweep(ground_normal_, axis_, yaw, pitch);
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
  get_if_present(
    node, "inner_prism_initial_theta", innerPrismTheta0, innerPrismTheta0);

  double innerPrismRot = 3.5;
  get_if_present(
    node, "inner_prism_rotation_rate", innerPrismRot, innerPrismRot);

  double innerPrismAzi = 15.2;
  get_if_present(node, "inner_prism_azimuth", innerPrismAzi, innerPrismAzi);

  double outerPrismTheta0 = 90;
  get_if_present(
    node, "outer_prism_initial_theta", outerPrismTheta0, outerPrismTheta0);

  double outerPrismRot = 6.5;
  get_if_present(
    node, "outer_prism_rotation_rate", outerPrismRot, outerPrismRot);

  double outerPrismAzi = 15.2;
  get_if_present(node, "outer_prism_azimuth", outerPrismAzi, outerPrismAzi);

  innerPrism_ = {convert::degrees_to_radians(innerPrismTheta0),
                 convert::rotations_to_radians(innerPrismRot),
                 convert::degrees_to_radians(innerPrismAzi)};
  outerPrism_ = {convert::degrees_to_radians(outerPrismTheta0),
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

namespace {

struct Triangle
{
  vs::Vector v0;
  vs::Vector v1;
  vs::Vector v2;
};

std::array<Triangle, 4> subdivide_face(std::array<vs::Vector, 4> face)
{
  std::array<Triangle, 4> tris;

  auto face_mid = 0.25 * (face[0] + face[1] + face[2] + face[3]);
  for (int itriangle = 0; itriangle < 4; ++itriangle) {
    const int base_vertex = itriangle;
    const int cyclic_next = (itriangle + 1) % 4;
    tris[itriangle].v0 = face[base_vertex];
    tris[itriangle].v1 = face[cyclic_next];
    tris[itriangle].v2 = face_mid;
  }
  return tris;
}

std::array<vs::Vector, 4>
  face_coordinates(std::array<vs::Vector, 8> box, int index)
{
  constexpr int vertex_face_table[6][4] = {{0, 1, 2, 3}, {1, 2, 6, 5},
                                           {4, 5, 6, 7}, {3, 2, 6, 7},
                                           {0, 3, 7, 4}, {0, 1, 5, 4}};
  std::array<vs::Vector, 4> face;
  for (int n = 0; n < 4; ++n) {
    face[n] = box[vertex_face_table[index][n]];
  }
  return face;
}

std::pair<bool, vs::Vector>
triangle_line_intersection(
  Triangle tri, vs::Vector origin, vs::Vector line, double tol)
{
  // trumbore-moller
  const auto edge1 = tri.v1 - tri.v0;
  const auto edge2 = tri.v2 - tri.v0;
  const auto h = line ^ edge2; // cross product
  const auto a = edge1 & h;    // dot product
  if (a > -tol && a < tol) {
    return {false, {}};
  }
  const auto f = 1.0 / a;
  const auto s = origin - tri.v0;
  const auto u = f * (h & s);
  if (u < 0.0 || u > 1.0) {
    return {false, {}};
  }
  const auto q = s ^ edge1;
  const auto v = f * (line & q);
  if (v < 0.0 || u + v > 1.0) {
    return {false, {}};
  }
  const auto t = f * (edge2 & q);
  if (t > tol) {
    return {true, origin + line * t};
  }
  return {false, {}};
}

std::array<double, 3>
to_array3(vs::Vector x)
{
  return {x[0], x[1], x[2]};
}

} // namespace

namespace details {
std::pair<bool, Segment> line_intersection_with_box(
  std::array<vs::Vector, 8> box, vs::Vector origin, vs::Vector line)
{
  // do one subdivision of the box into triangular sections
  // then do trumbore-moller on each triangle to determine intersection points
  // core assumption here is that the box faces are almost surely planar
  // so one subdivision is overkill to begin with.

  const double tol = 100 * std::numeric_limits<double>::epsilon();
  line.normalize();

  const double large_value = std::cbrt(std::numeric_limits<double>::max());
  constexpr int box_faces = 6;
  constexpr int tri_per_face = 4;
  constexpr int num_tri = box_faces * tri_per_face;

  vs::Vector large = origin + vs::Vector(large_value, large_value, large_value);
  std::array<vs::Vector, num_tri> intersection;
  std::fill(intersection.begin(), intersection.end(), large);
  std::array<bool, num_tri> intersected;
  std::fill(intersected.begin(), intersected.end(), false);

  for (int f = 0; f < box_faces; ++f) {
    auto triangles = subdivide_face(face_coordinates(box, f));
    for (int t = 0; t < tri_per_face; ++t) {
      const int id = tri_per_face * f + t;
      auto[found, intersect] =
        triangle_line_intersection(triangles[t], origin, line, tol);
      intersected[id] = found;
      if (found) {
        intersection[id] = intersect;
      }
    }
  }

  bool some_intersection = false;
  for (int j = 0; j < num_tri; ++j) {
    some_intersection |= intersected[j];
    if (intersected[j]) {
      break;
    }
  }
  if (!some_intersection) {
    return {false, {}};
  }

  // can potentially get multiple triangles intersecting the same point, if for
  // instance the line goes through triangle's edge
  vs::Vector i0 = large;
  for (int j = 0; j < num_tri; ++j) {
    if (!intersected[j]) {
      continue;
    }

    if (vs::mag(intersection[j] - origin) < vs::mag(i0 - origin)) {
      i0 = intersection[j];
    }
  }

  vs::Vector i1 = large;
  for (int j = 0; j < num_tri; ++j) {
    if (!intersected[j]) {
      continue;
    }

    if (
      vs::mag(i0 - intersection[j]) > tol &&
      vs::mag(intersection[j] - origin) < vs::mag(i1 - origin)) {
      i1 = intersection[j];
    }
  }

  if (vs::mag(i1 - large) < tol) {
    // maybe only one intersection, e.g. the line is tangent to the box
    i1 = i0;
  }
  return {true, {to_array3(i1), to_array3(i0)}};
}
} // namespace details

void
RadarSegmentGenerator::load(const YAML::Node& node)
{
  center_ = to_array3(node["center"].as<Coordinates>());

  axis_ = to_array3(node["axis"].as<Coordinates>());
  normalize_vec3(axis_.data());

  double sweep_angle_in_degrees = 20;
  get_if_present(
    node, "sweep_angle", sweep_angle_in_degrees, sweep_angle_in_degrees);
  ThrowRequireMsg(
    sweep_angle_in_degrees >= 0, "Sweep angle must be semipositive");
  sweep_angle_ = convert::degrees_to_radians(sweep_angle_in_degrees);

  double angular_speed = 30; // deg/s
  get_if_present(node, "angular_speed", angular_speed, angular_speed);
  angular_speed_ = convert::degrees_to_radians(angular_speed);

  beam_length_ = 50e3; // m
  get_required(node, "beam_length", beam_length_);

  if (node["ground_direction"]) {
    ground_normal_ = to_array3(node["ground_direction"].as<Coordinates>());
    normalize_vec3(ground_normal_.data());
  }

  auto box_corner_name = [](int n) { return "box_" + std::to_string(n); };
  if (node[box_corner_name(0)]) {
    throw std::runtime_error("box vertices are 1-indexed");
  }

  int specified_count = 0;
  for (int n = 0; n < 8; ++n) {
    if (node[box_corner_name(n + 1)]) {
      ++specified_count;
    }
  }

  get_if_present(
    node, "reset_time_delta", reset_time_delta_, reset_time_delta_);
  ThrowRequireMsg(
    reset_time_delta_ >= 0, "reset time delta must be semi-positive");

  if (specified_count > 0 && specified_count != 8) {
    throw std::runtime_error("Must specify entire box with `box_n` syntax");
  } else if (node["bbox"]) {
    auto bbox = node["bbox"].as<std::vector<double>>();
    if (bbox.size() != 6) {
      throw std::runtime_error("bbox specification requires six coordinates");
    }
    const auto dx =
      vs::Vector(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]);
    box_[0] = {bbox[0], bbox[1], bbox[2]};
    box_[1] = box_[0] + vs::Vector(dx[0], 0, 0);
    box_[2] = box_[0] + vs::Vector(dx[0], dx[1], 0);
    box_[3] = box_[0] + vs::Vector(0, dx[1], 0);
    box_[4] = box_[0] + vs::Vector(0, 0, dx[2]);
    box_[5] = box_[0] + vs::Vector(dx[0], 0, dx[2]);
    box_[6] = box_[0] + vs::Vector(dx[0], dx[1], dx[2]);
    box_[7] = box_[0] + vs::Vector(0, dx[1], dx[2]);
  } else {
    for (int n = 0; n < 8; ++n) {
      box_[n] = to_vec3(node[box_corner_name(n + 1)].as<Coordinates>());
    }
  }

  // basic check to see if there's some volume to the box
  // maybe make this a real volume calculation
  const auto dx = (box_[6] - box_[0]);
  const double small = 1e-15;
  if (dx[0] < small || dx[1] < small || dx[2] < small) {
    throw std::runtime_error("Box has no volume");
  }

  if (node["elevation_angles"]) {
    elevation_table_ = node["elevation_angles"].as<std::vector<double>>();
    std::transform(
      elevation_table_.cbegin(), elevation_table_.cend(),
      elevation_table_.begin(), convert::degrees_to_radians);
  }
}

double
RadarSegmentGenerator::total_sweep_time() const
{
  return 2 * (sweep_angle_ / angular_speed_ + reset_time_delta_);
}

double
RadarSegmentGenerator::periodic_time(double time) const
{
  /* radar overshoots the sweep angle, resets, and hits the constant
   angular speed before getting back into the sweep range.
   we're going to model this as just going to the end and instantly reversing
 */
  return time - std::floor(time / total_sweep_time()) * total_sweep_time();
}

int
RadarSegmentGenerator::sweep_count(double time) const
{
  const double sweep_time = sweep_angle_ / angular_speed_ + reset_time_delta_;
  return std::floor(time / sweep_time);
}

RadarSegmentGenerator::phase
RadarSegmentGenerator::determine_operation_phase(double periodic_time) const
{
  const double phase_time = sweep_angle_ / angular_speed_;

  if (periodic_time < phase_time) {
    return phase::FORWARD;
  } else if (periodic_time < phase_time + reset_time_delta_) {
    return phase::FORWARD_PAUSE;
  } else if (periodic_time < 2 * phase_time + reset_time_delta_) {
    return phase::REVERSE;
  } else {
    return phase::REVERSE_PAUSE;
  }
}

double
RadarSegmentGenerator::determine_current_angle(double periodic_time) const
{
  switch (determine_operation_phase(periodic_time)) {
  case phase::FORWARD: {
    return angular_speed_ * periodic_time - sweep_angle_ / 2;
  }
  case phase::FORWARD_PAUSE: {
    return sweep_angle_ / 2;
  }
  case phase::REVERSE: {
    return 3 * sweep_angle_ / 2 -
           angular_speed_ * (periodic_time - reset_time_delta_);
  }
  default: {
    return -sweep_angle_ / 2;
  }
  }
}

Segment
RadarSegmentGenerator::generate(double time) const
{
  /*
   radar is taken from an assumed far away location and sweeps through a set of
   angles at a fixed angular velocity, reversing itself at the end of its step.

   model is to clip out of the segment intersected some user defined box and
   keep a fixed number of points along a now varying length line segment.
   presumably the full domain of the simulation.  We don't require that box have
   planar faces but the intersection point isn't incredibly accurate for heavily
   skewed boxes.
  */

  const double clockwise = -1;
  const auto pitch = clockwise * elevation_table_.at(
                                   sweep_count(time) % elevation_table_.size());
  const auto tail = center_;
  const auto yaw = determine_current_angle(periodic_time(time));
  const auto sight_vector = plane_sweep(ground_normal_, axis_, yaw, pitch);
  const auto tip = affine(center_, beam_length_, sight_vector);

  auto[found, seg] = details::line_intersection_with_box(
    box_, to_vec3(center_), to_vec3(sight_vector));
  if (!found) {
    // return the unclipped line if it doesn't intersect the domain
    return {tip, tail, false};
  }
  const double tol = 100 * std::numeric_limits<double>::epsilon();
  if (found && vs::mag(to_vec3(seg.tail_) - to_vec3(seg.tip_)) < tol) {
    // found one match. consider it unmatched and return the base line
    return {tip, tail, false};
  }

  if (vs::mag(to_vec3(seg.tip_) - to_vec3(center_)) > beam_length_) {
    seg.tip_ = tip;
  }

  if (vs::mag(to_vec3(seg.tail_) - to_vec3(center_)) > beam_length_) {
    seg.tail_ = tail;
  }
  return {seg.tip_, seg.tail_, true};
}

} // namespace sierra::nalu
