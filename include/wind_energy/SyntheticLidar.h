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

#include "wind_energy/LidarPatterns.h"
#include "vs/vector.h"

#include <memory>
#include <array>
#include <optional>

namespace sierra {
namespace nalu {

struct RadarFilter
{
  std::vector<vs::Vector> rays;
  std::vector<double> weights;
};

class LidarLineOfSite
{
public:
  void load(const YAML::Node& node);
  std::unique_ptr<DataProbeSpecInfo>
  determine_line_of_site_info(const YAML::Node& node);

  double time() const { return lidar_time_; }

  void set_time(double t) { lidar_time_ = t; }
  void increment_time() { lidar_time_ += lidar_dt_; }

  void output(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& active,
    const std::string& coordinates_name,
    double dtratio);

  void output_cone_filtered(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& active,
    const std::string& coordinates_name,
    double dtratio);

private:
  enum class Output { NETCDF, TEXT, DATAPROBE } output_type_{Output::NETCDF};
  enum class Predictor {
    NEAREST,
    FORWARD_EULER
  } predictor_{Predictor::FORWARD_EULER};

  std::unique_ptr<SegmentGenerator> segGen;

  void prepare_nc_file();
  void output_nc(
    double time,
    const std::vector<std::array<double, 3>>& x,
    const std::vector<std::array<double, 3>>& u);
  void output_txt(
    double time,
    const std::vector<std::array<double, 3>>& x,
    const std::vector<std::array<double, 3>>& u,
    std::ofstream& file);
  void output_txt_los(
    double time,
    const std::vector<std::array<double, 3>>& x,
    const std::vector<double>& u,
    int n,
    std::ofstream& file);

  void output_nc_filtered(
    double time,
    const std::vector<std::array<double, 3>>& x,
    const std::vector<double>& u,
    int n);

  std::map<std::string, int> ncVarIDs_;

  std::optional<std::ofstream> file_ = {};

  mutable double lidar_time_{0};
  mutable size_t internal_output_counter_{0};
  std::unique_ptr<LocalVolumeSearchData> search_data_;

  double lidar_dt_{2. / 984};
  double scanTime_{2};
  int nsamples_{984};
  int npoints_{100};
  std::vector<std::string> fromTargetNames_;

  std::string name_{"lidar-los"};
  std::string fname_{"lidar-los.nc"};
  bool warn_on_missing_{false};
  bool reuse_search_data_{true};
  bool always_output_{false};
  RadarFilter radar_data_;
};

namespace details {
std::vector<vs::Vector>
make_radar_grid(double phi, int nphi, int ntheta, vs::Vector axis);

RadarFilter parse_radar_filter(const YAML::Node& node);

std::pair<std::vector<vs::Vector>, std::vector<double>> spherical_cap_radau(
  double gammav,
  int ntheta,
  int nphi,
  std::function<double(double)> f = nullptr);

enum class NormalRule { SIGMA1, SIGMA2, SIGMA3, HALFPOWER };

std::pair<std::vector<vs::Vector>, std::vector<double>>
spherical_cap_truncated_normal(double gammav, int ntheta, NormalRule rule);

} // namespace details

class LidarLOS
{
public:
  void output(
    const stk::mesh::BulkData& bulk,
    const stk::mesh::Selector& sel,
    const std::string& coords_name,
    double dt,
    double time);

  void load(const YAML::Node& node, DataProbePostProcessing* probes);
  void set_time_for_all(double time);
  bool start_time_has_been_set() { return start_time_has_been_set_; };

private:
  bool start_time_has_been_set_{false};
  std::vector<LidarLineOfSite> lidars_;
  std::vector<LidarLineOfSite> radars_;
};

} // namespace nalu
} // namespace sierra

#endif
