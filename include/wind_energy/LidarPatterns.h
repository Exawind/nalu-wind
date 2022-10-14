// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LidarPatterns_H
#define LidarPatterns_H

#include "vs/vector.h"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

struct Segment
{
  std::array<double, 3> tip_{};
  std::array<double, 3> tail_{};
};

class SegmentGenerator
{
public:
  virtual ~SegmentGenerator() = default;
  virtual void load(const YAML::Node& node) = 0;
  virtual Segment generate(double time) const = 0;
};

enum class SegmentType { SPINNER, SCANNING, RADAR };
SegmentType segment_generator_types(std::string name);
std::unique_ptr<SegmentGenerator> make_segment_generator(SegmentType type);
std::unique_ptr<SegmentGenerator>
make_segment_generator(const std::string& name);

namespace convert {
inline double
degrees_to_radians(double deg)
{
  return (M_PI / 180) * deg;
}
inline double
rotations_to_radians(double rot)
{
  return (2 * M_PI) * rot;
}
} // namespace convert

class ScanningLidarSegmentGenerator final : public SegmentGenerator
{
public:
  void load(const YAML::Node& node) final;
  Segment generate(double t) const final;

private:
  enum class phase { FORWARD, RESET };
  double periodic_time(double time) const;
  phase determine_operation_phase(double periodic_time) const;
  double angle_if_during_reset(double periodic_time) const;
  double angle_if_during_forward_phase(double periodic_time) const;
  double determine_current_angle(double periodic_time) const;
  double determine_end_of_forward_phase() const
  {
    return (sweep_angle_ / step_delta_angle_) * stare_time_;
  }

  int periodic_count(double time) const;
  double determine_elevation_angle(int sweep_count) const;

  double end_of_forward_phase_{determine_end_of_forward_phase()};

  std::vector<double> elevation_table_{0};
  double beam_length_{1.0};
  std::array<double, 3> center_{0, 0, 0};
  std::array<double, 3> axis_{1, 0, 0};
  std::array<double, 3> ground_normal_{0, 0, 1};
  double sweep_angle_{convert::degrees_to_radians(20)};
  double step_delta_angle_{convert::degrees_to_radians(1)};
  double stare_time_{1};
  double reset_time_delta_{1};

  enum class direction {
    CLOCKWISE = -1,
    CCLOCKWISE = 1
  } dir_{direction::CLOCKWISE};
};

class SpinnerLidarSegmentGenerator final : public SegmentGenerator
{
public:
  void load(const YAML::Node& node) final;
  Segment generate(double time) const final;

private:
  double beamLength_{1.0};
  std::array<double, 3> lidarCenter_{0, 0, 0};
  std::array<double, 3> laserAxis_{1, 0, 0};
  std::array<double, 3> groundNormal_{0, 0, 1};

  struct PrismParameters
  {
    double theta(double time) const { return theta0_ + rot_ * time; }

    double theta0_{0};  // rad
    double rot_{0};     // rad / s
    double azimuth_{0}; // rad
  };
  PrismParameters innerPrism_{
    convert::degrees_to_radians(90), convert::rotations_to_radians(3.5),
    convert::degrees_to_radians(15.2)};
  PrismParameters outerPrism_{
    convert::degrees_to_radians(90), convert::rotations_to_radians(6.5),
    convert::degrees_to_radians(15.2)};
};

class RadarSegmentGenerator final : public SegmentGenerator
{
public:
  void load(const YAML::Node& node) final;
  Segment generate(double t) const final;
  void set_axis(vs::Vector axis);

private:
  enum class phase { FORWARD, FORWARD_PAUSE, REVERSE, REVERSE_PAUSE };
  double periodic_time(double time) const;
  int periodic_count(double time) const;
  phase determine_operation_phase(double periodic_time) const;
  double determine_current_angle(double periodic_time) const;
  double determine_elevation_angle(int sweep_count) const;
  double total_sweep_time() const;
  double reset_time_delta_{1.0};
  double sweep_angle_{convert::degrees_to_radians(20)};
  double angular_speed_{convert::degrees_to_radians(30)}; // per second
  double beam_length_{1.0};
  std::array<double, 3> center_{0, 0, 0};
  std::array<double, 3> axis_{1, 0, 0};
  std::array<double, 3> ground_normal_{0, 0, 1};
  std::array<vs::Vector, 8> box_;
  std::vector<double> elevation_table_{0};
};

namespace details {
std::pair<bool, Segment> line_intersection_with_box(
  std::array<vs::Vector, 8> box, vs::Vector origin, vs::Vector line);
}

} // namespace nalu
} // namespace sierra

#endif
