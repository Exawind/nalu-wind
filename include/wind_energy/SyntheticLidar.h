// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SyntheticLidar_H
#define SyntheticLidar_H

#include <DataProbePostProcessing.h>

#include "xfer/LocalVolumeSearch.h"

#include <memory>
#include <array>

namespace sierra {
namespace nalu {

struct Segment
{
  Segment() = default;
  Segment(std::array<double, 3> tip, std::array<double, 3> tail)
    : tip_(tip), tail_(tail){};

  std::array<double, 3> tip_{{}};
  std::array<double, 3> tail_{{}};
};

struct PrismParameters
{
  PrismParameters() = default;
  PrismParameters(double theta0, double rot, double azimuth)
    : theta0_{theta0}, rot_{rot}, azimuth_{azimuth} {};

  double theta0_{0};  // rad
  double rot_{0};     // rad / s
  double azimuth_{0}; // rad
};

class SpinnerLidarSegmentGenerator
{
public:
  SpinnerLidarSegmentGenerator() = default;

  SpinnerLidarSegmentGenerator(
    PrismParameters inner, PrismParameters outer, double in_beamLength);

  void load(const YAML::Node& node);

  Segment generate_path_segment(double time) const;

  void set_inner_prism(PrismParameters innerPrism) { innerPrism_ = innerPrism; }
  void set_inner_prism(double theta0, double rot, double azi)
  {
    innerPrism_ = {theta0, rot, azi};
  }
  void set_outer_prism(PrismParameters outerPrism) { outerPrism_ = outerPrism; }
  void set_outer_prism(double theta0, double rot, double azi)
  {
    outerPrism_ = {theta0, rot, azi};
  }
  void set_lidar_center(Coordinates lidarCenter)
  {
    lidarCenter_ = {{lidarCenter.x_, lidarCenter.y_, lidarCenter.z_}};
  }
  void set_laser_axis(Coordinates laserAxis)
  {
    laserAxis_ = {{laserAxis.x_, laserAxis.y_, laserAxis.z_}};
  }
  void set_ground_normal(Coordinates gNormal)
  {
    groundNormal_ = {{gNormal.x_, gNormal.y_, gNormal.z_}};
  }
  void set_beam_length(double beamLength) { beamLength_ = beamLength; }

private:
  PrismParameters innerPrism_;
  PrismParameters outerPrism_;
  double beamLength_{1.0};

  std::array<double, 3> lidarCenter_{{0, 0, 0}};
  std::array<double, 3> laserAxis_{{1, 0, 0}};
  std::array<double, 3> groundNormal_{{0, 0, 1}};
};

class LidarLineOfSite
{
public:
  LidarLineOfSite() = default;
  void load(const YAML::Node& node);

  std::unique_ptr<DataProbeSpecInfo>
  determine_line_of_site_info(const YAML::Node& node);

  double time() { return lidar_time_; }
  void set_time(double t) { lidar_time_ = t; }
  void increment_time() { lidar_time_ += lidar_dt_; }

  void output(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& active,
    const std::string& coordinates_name,
    double dtratio);

private:
  enum class Output { NETCDF, TEXT, DATAPROBE } output_type_{Output::NETCDF};
  SpinnerLidarSegmentGenerator segGen;

  void prepare_nc_file();
  void output_nc(
    double time,
    const std::vector<std::array<double, 3>>& x,
    const std::vector<std::array<double, 3>>& u);
  void output_txt(
    double time,
    const std::vector<std::array<double, 3>>& x,
    const std::vector<std::array<double, 3>>& u);
  std::map<std::string, int> ncVarIDs_;

  mutable double lidar_time_{0};
  mutable size_t internal_output_counter_{0};
  std::unique_ptr<LocalVolumeSearchData> search_data_;

  double lidar_dt_{2. / 984};
  double scanTime_{2};
  int nsamples_{984};
  int npoints_{100};
  std::vector<std::string> fromTargetNames_;

  std::string name_{"lidar-los"};
};

} // namespace nalu
} // namespace sierra

#endif
