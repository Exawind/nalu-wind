// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "wind_energy/SyntheticLidar.h"

#include "NaluParsing.h"
#include "master_element/TensorOps.h"

#include "xfer/Transfer.h"
#include "xfer/LocalVolumeSearch.h"
#include "netcdf.h"
#include "Ioss_FileInfo.h"

#include "vs/vector.h"
#include "vs/tensor.h"

#include <memory>

namespace sierra {
namespace nalu {

constexpr int dim = 3;

void
LidarLineOfSite::load(const YAML::Node& node)
{
  if (node["type"]) {
    segGen = make_segment_generator(node["type"].as<std::string>());
  } else {
    segGen = make_segment_generator(SegmentType::SPINNER);
  }

  if (node["radar_cone_filter"]) {
    radar_data_ = details::parse_radar_filter(node);
  }

  if (node["output"]) {
    const auto type = node["output"].as<std::string>();
    if (type == "text") {
      output_type_ = Output::TEXT;
      file_ = std::ofstream{};
    } else if (type == "netcdf") {
      output_type_ = Output::NETCDF;
    } else if ("dataprobes") {
      output_type_ = Output::DATAPROBE;
    }

    if (output_type_ != Output::TEXT && node["radar_cone_filter"]) {
      throw std::runtime_error(
        "Only text csv output implemented for radar filtering");
    }
  }

  get_required(node, "points_along_line", npoints_);
  get_if_present(node, "warn_on_missing", warn_on_missing_, warn_on_missing_);
  get_if_present(
    node, "reuse_search_data", reuse_search_data_, reuse_search_data_);
  get_if_present(node, "always_output", always_output_, always_output_);

  if (node["name"]) {
    name_ = node["name"].as<std::string>();
  }

  if (node["time_step"] && output_type_ != Output::DATAPROBE) {
    lidar_dt_ = node["time_step"].as<double>();
  } else if (node["frequency"] && output_type_ != Output::DATAPROBE) {
    lidar_dt_ = 1.0 / node["frequency"].as<double>();
  } else {
    get_required(node, "scan_time", scanTime_);
    get_required(node, "number_of_samples", nsamples_);
    lidar_dt_ = scanTime_ / nsamples_;
  }

  if (node["predictor"]) {
    const auto spec = node["predictor"].as<std::string>();
    std::map<std::string, Predictor> valid = {
      {"forward_euler", Predictor::FORWARD_EULER},
      {"nearest", Predictor::NEAREST}};
    if (valid.find(spec) != valid.end()) {
      predictor_ = valid.at(spec);
    } else {
      std::string valid_keys = "";
      for (const auto& pair : valid) {
        valid_keys += pair.first + " ";
      }
      valid_keys = valid_keys.substr(0, valid_keys.size() - 1);
      throw std::runtime_error(
        "invalid predictor spec: " + spec + ", valid specs are: " + valid_keys);
    }
  }

  const YAML::Node fromTargets = node["from_target_part"];
  if (fromTargets) {
    if (fromTargets.Type() == YAML::NodeType::Scalar) {
      fromTargetNames_.push_back(fromTargets.as<std::string>());
    } else {
      for (const auto& target : fromTargets) {
        fromTargetNames_.push_back(target.as<std::string>());
      }
    }
  }

  if (node["scanning_lidar_specifications"]) {
    segGen->load(node["scanning_lidar_specifications"]);
  } else if (node["radar_specifications"]) {
    segGen->load(node["radar_specifications"]);
  } else {
    segGen->load(node);
  }
}

bool
is_root(MPI_Comm comm, int root)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank == root;
}
void
check_nc_error(int code)
{

  if (code != 0) {
    throw std::runtime_error(
      "SyntheticLidar NetCDF error: " + std::string(nc_strerror(code)));
  }
}
namespace {
std::string
determine_filename(const std::string& name, const std::string& suffix)
{
  std::string fname = name + suffix;

  Ioss::FileInfo info(fname);
  if (info.exists()) {
    // give a large, finite amount of names to check
    const int max_restarts = 2048;
    bool found_valid = false;
    for (int j = 1; j < max_restarts; ++j) {
      fname = name + "-rst-" + std::to_string(j) + suffix;
      Ioss::FileInfo info_j(fname);
      if (!info_j.exists()) {
        found_valid = true;
        break;
      }
    }
    if (!found_valid) {
      throw std::runtime_error("Too many restarts checked");
    }
  }
  return fname;
}
} // namespace

void
LidarLineOfSite::prepare_nc_file()
{
  int ncid;

  fname_ = determine_filename(name_, ".nc");

  int ierr = nc_create(fname_.c_str(), NC_CLOBBER, &ncid);
  check_nc_error(ierr);

  // Define dimensions for the NetCDF file
  int tDim;
  ierr = nc_def_dim(ncid, "num_timesteps", NC_UNLIMITED, &tDim);
  check_nc_error(ierr);

  int pDim;
  ierr = nc_def_dim(ncid, "num_points", npoints_, &pDim);
  check_nc_error(ierr);

  int vDim;
  ierr = nc_def_dim(ncid, "vec_dim", 3, &vDim);
  check_nc_error(ierr);

  const int vec_dim[3] = {tDim, pDim, vDim};

  {
    int varid;
    ierr = nc_def_var(ncid, "step", NC_INT, 1, &tDim, &varid);
    check_nc_error(ierr);
    ncVarIDs_["step"] = varid;
  }

  auto add_ncvar = [&](std::string name, int dim, const int* const dims) {
    int varid;
    ierr = nc_def_var(ncid, name.c_str(), NC_DOUBLE, dim, dims, &varid);
    check_nc_error(ierr);
    ncVarIDs_[name] = varid;
  };

  add_ncvar("time", 1, &tDim);
  add_ncvar("coordinates", 3, vec_dim);
  add_ncvar("velocity", 3, vec_dim);

  // Indicate that we are done defining variables, ready to write data
  ierr = nc_enddef(ncid);
  check_nc_error(ierr);

  ierr = nc_close(ncid);
  check_nc_error(ierr);
}

void
LidarLineOfSite::output_nc(
  double time,
  const std::vector<std::array<double, 3>>& x,
  const std::vector<std::array<double, 3>>& u)
{
  if (internal_output_counter_ == 0) {
    prepare_nc_file();
  }

  int ncid, ierr;
  ierr = nc_open(fname_.c_str(), NC_WRITE, &ncid);
  check_nc_error(ierr);

  size_t scalar = 1;
  const size_t vector_list_start[] = {internal_output_counter_, 0, 0};
  const size_t vector_list_count[] = {1, static_cast<size_t>(npoints_), 3};

  const int step = static_cast<int>(internal_output_counter_);
  ierr = nc_put_vara_int(
    ncid, ncVarIDs_["step"], &internal_output_counter_, &scalar, &step);
  check_nc_error(ierr);

  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["time"], &internal_output_counter_, &scalar, &time);
  check_nc_error(ierr);

  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["coordinates"], vector_list_start, vector_list_count,
    &x[0][0]);
  check_nc_error(ierr);

  ierr = nc_put_vara_double(
    ncid, ncVarIDs_["velocity"], vector_list_start, vector_list_count,
    &u[0][0]);
  check_nc_error(ierr);

  ierr = nc_close(ncid);
  check_nc_error(ierr);

  ++internal_output_counter_;
}

void
LidarLineOfSite::output_txt(
  double time,
  const std::vector<std::array<double, 3>>& x,
  const std::vector<std::array<double, 3>>& u,
  std::ofstream& file)
{
  if (internal_output_counter_ == 0) {
    fname_ = determine_filename(name_, ".txt");
    file.exceptions(file.exceptions() | std::ios::failbit);
    file.open(fname_, std::ios::app);
    if (file.fail()) {
      throw std::ios_base::failure(std::strerror(errno));
    }
    file << "t,x,y,z,u,v,w" << std::endl;
  }
  for (size_t j = 0; j < x.size(); ++j) {
    file << std::setprecision(15) << time << "," << x.at(j)[0] << ","
         << x.at(j)[1] << "," << x.at(j)[2] << "," << u.at(j)[0] << ","
         << u.at(j)[1] << "," << u.at(j)[2] << "\n";
  }
  ++internal_output_counter_;
}

void
LidarLineOfSite::output_txt_los(
  double time,
  const std::vector<std::array<double, 3>>& x,
  const std::vector<double>& u_dot_l,
  int npoints,
  std::ofstream& file)
{
  if (internal_output_counter_ == 0) {
    fname_ = determine_filename(name_, ".txt");
    file.exceptions(file.exceptions() | std::ios::failbit);
    file.open(fname_, std::ios::app);
    if (file.fail()) {
      throw std::ios_base::failure(std::strerror(errno));
    }
    file << "t,x,y,z,u_dot_l" << std::endl;
    file << std::setprecision(15);
  }

  const int stride = x.size() / npoints;
  for (int j = 0; j < npoints; ++j) {
    const auto& pos = x[stride * j];
    file << time << "," << pos[0] << "," << pos[1] << "," << pos[2] << ","
         << u_dot_l.at(j) << "\n";
  }
  ++internal_output_counter_;
}

namespace {
vs::Vector
to_vec3(const std::array<double, 3>& x)
{
  return {x[0], x[1], x[2]};
}
} // namespace
void
LidarLineOfSite::output(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active,
  const std::string& coordinates_name,
  double dtratio)
{
  if (output_type_ == Output::DATAPROBE) {
    return;
  }

  const auto seg = segGen->generate(time());
  if (!seg.valid && !always_output_) {
    return;
  }

  if (!search_data_) {
    search_data_ =
      std::make_unique<LocalVolumeSearchData>(bulk, active, npoints_);
  }

  // segment length can shrink to zero, so mag(dx) isn't bounded from below
  const std::array<double, 3> dx{
    {(seg.tip_[0] - seg.tail_[0]) / (npoints_ > 1 ? (npoints_ - 1) : 1),
     (seg.tip_[1] - seg.tail_[1]) / (npoints_ > 1 ? (npoints_ - 1) : 1),
     (seg.tip_[2] - seg.tail_[2]) / (npoints_ > 1 ? (npoints_ - 1) : 1)}};

  std::vector<std::array<double, 3>> points(npoints_);
  for (int j = 0; j < npoints_; ++j) {
    points[j] = {
      {seg.tail_[0] + j * dx[0], seg.tail_[1] + j * dx[1],
       seg.tail_[2] + j * dx[2]}};
  }
  const auto& coord_field =
    *bulk.mesh_meta_data()
       .get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
         stk::topology::NODE_RANK, coordinates_name);

  const auto& velocity_field =
    bulk.mesh_meta_data()
      .get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "velocity")
      ->field_of_state(stk::mesh::StateNP1);

  const auto& velocity_prev =
    bulk.mesh_meta_data()
      .get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "velocity")
      ->field_of_state(stk::mesh::StateN);

  const double extrap_dt = predictor_ == Predictor::NEAREST ? 0 : dtratio;
  local_field_interpolation(
    bulk, active, points, coord_field, velocity_prev, velocity_field, extrap_dt,
    *search_data_);

  const auto& lcl_velocity = search_data_->interpolated_values;
  const auto& lcl_ownership = search_data_->ownership;

  auto comm = bulk.parallel();
  const int root = 0;

  std::vector<std::array<double, 3>> velocity(npoints_, {0, 0, 0});
  MPI_Reduce(
    lcl_velocity.data(), velocity.data(), 3 * npoints_, MPI_DOUBLE, MPI_SUM,
    root, comm);

  // parallel reconciliation for points along processor boundaries is to
  // do an arithmetic average, assuming continuity.
  std::vector<int> degree(npoints_, 0);
  MPI_Reduce(
    lcl_ownership.data(), degree.data(), npoints_, MPI_INT, MPI_SUM, root,
    comm);

  if (is_root(comm, root)) {
    int not_found_count = 0;

    std::array<double, dim> max_unmatched{
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::lowest()};
    std::array<double, dim> min_unmatched{
      std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max()};
    for (int j = 0; j < npoints_; ++j) {
      const auto degj = degree.at(j);
      if (degj == 0) {
        ++not_found_count;
        for (int d = 0; d < 3; ++d) {
          max_unmatched[d] = std::max(max_unmatched[d], points.at(j)[d]);
          min_unmatched[d] = std::min(min_unmatched[d], points.at(j)[d]);
        }
      }
      const double inv_deg =
        (degj > 0) ? 1 / static_cast<double>(degree[j]) : 0;
      for (int d = 0; d < 3; ++d) {
        velocity.at(j)[d] *= inv_deg;
      }
    }
    if (not_found_count > 0 && warn_on_missing_) {

      auto lidar_name_start = name_.find_last_of("/");
      auto lidar_name = name_.substr(lidar_name_start + 1);

      NaluEnv::self().naluOutputP0()
        << "LIDAR " << lidar_name << " search did not match " << not_found_count
        << " points, max individually unmatched coords: (" << max_unmatched[0]
        << ", " << max_unmatched[1] << ", " << max_unmatched[2] << ")"
        << ", min individually unmatched coords: (" << min_unmatched[0] << ", "
        << min_unmatched[1] << ", " << min_unmatched[2] << ")" << std::endl;
    }

    if (not_found_count == npoints_ && !always_output_) {
      return;
    }

    if (internal_output_counter_ == 0) {
      Ioss::FileInfo::create_path(name_);
    }

    if (output_type_ == Output::TEXT) {
      std::vector<double> ulos(velocity.size());
      vs::Vector ray = to_vec3(dx);
      ray.normalize();
      for (size_t j = 0; j < velocity.size(); ++j) {
        ulos[j] = ray & to_vec3(velocity[j]);
      }
      output_txt_los(time(), points, ulos, points.size(), *file_);
    } else if (output_type_ == Output::NETCDF) {
      output_nc(time(), points, velocity);
    }
  }
  if (!reuse_search_data_) {
    search_data_.reset();
  }
}

void
line_average(
  const std::vector<int>& degree,
  const std::vector<double>& weights,
  const std::vector<double>& values,
  std::vector<double>& reduced)
{
  // occasionally points will be missing from the sum, if the point is outside
  // the simulation box. We could wait until all points are in the domain
  // to start counting.  Erring on the side of overreporting, we hope the
  // partial sum is a reasonable estimate of the average value, albeit
  // biased spatially

  const int nline = static_cast<int>(values.size() / weights.size());
  const int nquad = static_cast<int>(weights.size());
  for (int n = 0; n < nline; ++n) {
    double weight_sum = 0;
    double average = 0;
    for (int j = 0; j < nquad; ++j) {
      const int point_idx = nquad * n + j;
      const int inv_deg = degree[point_idx] > 0 ? 1. / degree[point_idx] : 0;
      weight_sum += int(degree[point_idx] > 0) * weights[j];
      average += weights[j] * values[point_idx] * inv_deg;
    }
    if (weight_sum > 0) {
      reduced[n] = average / weight_sum;
    } else {
      reduced[n] = 0;
    }
  }
}

namespace {
vs::Tensor
skew_cross(vs::Vector a, vs::Vector b)
{
  auto cross = b ^ a;
  return vs::Tensor(
    0, -cross[2], cross[1], cross[2], 0, -cross[0], -cross[1], cross[0], 0);
}

vs::Tensor
scale(vs::Tensor v, double a)
{
  vs::Tensor vnew;
  for (int j = 0; j < 9; ++j) {
    vnew[j] = a * v[j];
  }
  return vnew;
}

vs::Tensor
rotation_matrix(vs::Vector dst, vs::Vector src)
{
  auto vmat = skew_cross(dst, src);
  const auto ang = dst & src;

  const double small = 1e-14 * vs::mag(dst);
  if (std::abs(1 + ang) < small) {
    return scale(vs::Tensor::I(), -1);
  }
  return vs::Tensor::I() + vmat + scale((vmat & vmat), 1. / (1 + ang));
}

} // namespace

void
LidarLineOfSite::output_cone_filtered(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& active,
  const std::string& coordinates_name,
  double dtratio)
{
  const auto* radar = dynamic_cast<RadarSegmentGenerator*>(segGen.get());
  ThrowRequire(radar);

  auto seg = radar->generate(time());
  if (!(seg.valid || always_output_)) {
    return;
  }
  const auto center = radar->center();
  auto line_vector = vs::Vector(seg.tip_[0], seg.tip_[1], seg.tip_[2]) - center;
  line_vector.normalize();

  const auto& weights = radar_data_.weights;
  const auto& rays = radar_data_.rays;

  const auto dn = (npoints_ - 1);
  ThrowRequireMsg(dn >= 0, "At least two points required");

  const vs::Vector dx(
    (seg.tip_[0] - seg.tail_[0]) / dn, (seg.tip_[1] - seg.tail_[1]) / dn,
    (seg.tip_[2] - seg.tail_[2]) / dn);

  const int nquad = int(radar_data_.rays.size());
  if (!search_data_) {
    search_data_ =
      std::make_unique<LocalVolumeSearchData>(bulk, active, nquad * npoints_);
  }

  std::vector<std::array<double, 3>> points(nquad * npoints_);

  const auto canon_vector = vs::Vector(0, 0, 1);
  const auto transform = rotation_matrix(line_vector, canon_vector);
  for (int n = 0; n < npoints_; ++n) {
    vs::Vector axis_point(
      seg.tail_[0] + n * dx[0], seg.tail_[1] + n * dx[1],
      seg.tail_[2] + n * dx[2]);
    for (int j = 0; j < nquad; ++j) {
      const auto radius = vs::mag(axis_point - center);
      const auto point = radius * (transform & rays[j]) + center;
      points[nquad * n + j] = {point[0], point[1], point[2]};
    }
  }

  const auto& coord_field =
    *bulk.mesh_meta_data()
       .get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
         stk::topology::NODE_RANK, coordinates_name);

  const auto& velocity_field =
    bulk.mesh_meta_data()
      .get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "velocity")
      ->field_of_state(stk::mesh::StateNP1);

  const auto& velocity_prev =
    bulk.mesh_meta_data()
      .get_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "velocity")
      ->field_of_state(stk::mesh::StateN);

  const double extrap_dt = predictor_ == Predictor::NEAREST ? 0 : dtratio;

  local_field_interpolation(
    bulk, active, points, coord_field, velocity_prev, velocity_field, extrap_dt,
    *search_data_);
  auto& lcl_velocity = search_data_->interpolated_values;
  auto& lcl_ownership = search_data_->ownership;

  auto comm = bulk.parallel();
  const int root = 0;

  std::vector<double> lcl_line_velocity(nquad * npoints_, 0);
  for (int j = 0; j < nquad; ++j) {
    auto ray = transform & rays[j];
    ray.normalize();
    for (int n = 0; n < npoints_; ++n) {
      lcl_line_velocity[nquad * n + j] =
        ray & to_vec3(lcl_velocity[nquad * n + j]);
    }
  }

  std::vector<double> line_velocity(nquad * npoints_, 0);
  MPI_Reduce(
    lcl_line_velocity.data(), line_velocity.data(), line_velocity.size(),
    MPI_DOUBLE, MPI_SUM, root, comm);
  std::vector<int> degree(nquad * npoints_, 0);
  MPI_Reduce(
    lcl_ownership.data(), degree.data(), degree.size(), MPI_INT, MPI_SUM, root,
    comm);

  if (is_root(comm, root)) {
    int not_found_count = 0;
    for (int n = 0; n < npoints_ * nquad; ++n) {
      not_found_count += static_cast<int>(degree[n] == 0);
    }
    if (not_found_count != npoints_ * nquad) {
      std::vector<double> avg_line_velocity(npoints_, 0);
      line_average(degree, weights, line_velocity, avg_line_velocity);
      if (internal_output_counter_ == 0) {
        Ioss::FileInfo::create_path(name_);
      }
      // only text for now, check at parse
      ThrowRequire(output_type_ == Output::TEXT);
      output_txt_los(time(), points, avg_line_velocity, npoints_, *file_);
    }
  }
}

std::unique_ptr<DataProbeSpecInfo>
LidarLineOfSite::determine_line_of_site_info(const YAML::Node& node)
{
  load(node);

  auto lidarLOSInfo = std::make_unique<DataProbeSpecInfo>();

  lidarLOSInfo->xferName_ = "LidarSampling_xfer";
  lidarLOSInfo->fromToName_.emplace_back("velocity", "velocity_probe");
  lidarLOSInfo->fieldInfo_.emplace_back("velocity_probe", 3);
  lidarLOSInfo->fromTargetNames_ = fromTargetNames_;

  auto probeInfo = std::make_unique<DataProbeInfo>();

  probeInfo->isLineOfSite_ = true;
  probeInfo->numProbes_ = nsamples_;
  probeInfo->partName_.resize(nsamples_);
  probeInfo->processorId_.resize(nsamples_);
  probeInfo->numPoints_.resize(nsamples_);
  probeInfo->generateNewIds_.resize(nsamples_);
  probeInfo->tipCoordinates_.resize(nsamples_);
  probeInfo->tailCoordinates_.resize(nsamples_);
  probeInfo->nodeVector_.resize(nsamples_);
  probeInfo->part_.resize(nsamples_);
  probeInfo->geomType_.resize(nsamples_);

  const int numProcs = NaluEnv::self().parallel_size();
  const int divProcProbe = std::max(numProcs / nsamples_, numProcs);

  for (int ilos = 0; ilos < nsamples_; ilos++) {
    const double lidarTime = scanTime_ / (double)nsamples_ * ilos;
    Segment seg = segGen->generate(lidarTime);

    probeInfo->processorId_[ilos] = divProcProbe > 0 ? ilos % divProcProbe : 0;
    probeInfo->partName_[ilos] = name_ + "_" + std::to_string(ilos);
    probeInfo->numPoints_[ilos] = npoints_;
    probeInfo->geomType_[ilos] = DataProbeGeomType::LINEOFSITE;

    probeInfo->tipCoordinates_[ilos].x_ = seg.tip_[0];
    probeInfo->tipCoordinates_[ilos].y_ = seg.tip_[1];
    probeInfo->tipCoordinates_[ilos].z_ = seg.tip_[2];

    probeInfo->tailCoordinates_[ilos].x_ = seg.tail_[0];
    probeInfo->tailCoordinates_[ilos].y_ = seg.tail_[1];
    probeInfo->tailCoordinates_[ilos].z_ = seg.tail_[2];
  }
  lidarLOSInfo->dataProbeInfo_.push_back(probeInfo.release());

  return lidarLOSInfo;
}

void
LidarLOS::set_time_for_all(double time)
{
  for (auto& los : lidars_) {
    los.set_time(time);
  }
  for (auto& los : radars_) {
    los.set_time(time);
  }
  start_time_has_been_set_ = true;
}
namespace details {
std::pair<std::vector<vs::Vector>, std::vector<double>>
spherical_cap_quadrature(
  double gammav,
  int ntheta,
  std::vector<double> abscissae1D,
  std::vector<double> weights1D)
{
  // tensor-product quadrature. theta is uniform on [0, 2 pi) since it's a
  // circle and phi is a standard 1D quadrature of choice, mapped to
  // [cos(gamma), 1] with a variable transformation, following along from
  // DOI 10.1007/s10444-011-9187-2
  std::transform(
    abscissae1D.cbegin(), abscissae1D.cend(), abscissae1D.begin(),
    [gammav](double s) {
      return 0.5 * (1 - std::cos(gammav)) * (-s) + 0.5 * (1 + std::cos(gammav));
    });
  std::transform(
    weights1D.cbegin(), weights1D.cend(), weights1D.begin(),
    [gammav](double w) { return w * 0.5 * (1 - std::cos(gammav)); });

  std::vector<vs::Vector> rays;
  rays.reserve(abscissae1D.size() * ntheta);
  std::vector<double> weights;
  weights.reserve(rays.size());

  const int nphi = int(abscissae1D.size());

  // avoid ntheta multiplicity at the center
  rays.push_back(vs::Vector(0, 0, 1));
  weights.push_back(2 * M_PI * weights1D[0]);

  const auto theta_weight = 2 * M_PI / ntheta;
  for (int j = 1; j < nphi; ++j) {
    const auto tau = abscissae1D[j];
    for (int i = 0; i < ntheta; ++i) {
      const double theta = (2 * M_PI / ntheta) * i;
      const auto xr = std::sqrt(1 - tau * tau) * std::cos(theta);
      const auto yr = std::sqrt(1 - tau * tau) * std::sin(theta);
      const auto zr = tau;
      auto ray = vs::Vector(xr, yr, zr);
      ray.normalize();
      rays.push_back(ray);
      const auto phi_weight = weights1D[j];
      weights.push_back(theta_weight * phi_weight);
    }
  }
  return {rays, weights};
}

std::pair<std::vector<double>, std::vector<double>>
radau_rule(int n)
{
  // only include the positive half of the symmetric quadrature
  switch (n) {
  case 1:
    return {{-1}, {2}};
  case 2:
    return {{-1, -1. / 3}, {0.5, 1.5}};
  case 3:
    return {
      {-1, -0.28989794855663562, 0.68989794855663562},
      {1. / 9, 1.0249716523768432, 0.75280612540093455}};
  case 4:
    return {
      {-1.0000000000000000, -0.57531892352169411, 0.18106627111853058,
       0.82282408097459211},
      {0.12500000000000000, 0.65768863996011949, 0.77638693768634376,
       0.44092442235353675}};
  case 5:
    return {
      {-1.0000000000000000, -0.72048027131243890, -0.16718086473783364,
       0.44631397272375234, 0.88579160777096464},
      {0.080000000000000000, 0.44620780216714149, 0.62365304595148251,
       0.56271203029892412, 0.28742712158245188}};
  case 6:
    return {
      {-1.0000000000000000, -0.80292982840234715, -0.39092854670727219,
       0.12405037950522771, 0.60397316425278365, 0.92038028589706252},
      {0.055555555555555556, 0.31964075322051097, 0.48538718846896992,
       0.52092678318957498, 0.41690133431190774, 0.20158838525348084}};
  case 7:
    return {
      {-1.0000000000000000, -0.85389134263948223, -0.53846772406010900,
       -0.11734303754310026, 0.32603061943769140, 0.70384280066303142,
       0.94136714568043022},
      {0.040816326530612245, 0.23922748922531241, 0.38094987364423115,
       0.44710982901456647, 0.42470377900595561, 0.31820423146730148,
       0.14898847111202064}};
  case 8:
    return {
      {-1.0000000000000000, -0.88747487892615571, -0.63951861652621527,
       -0.29475056577366073, 0.094307252661110766, 0.46842035443082106,
       0.77064189367819154, 0.95504122712257500},
      {0.031250000000000000, 0.18535815480297928, 0.30413062064678513,
       0.37651754538911856, 0.39157216745249359, 0.34701479563450128,
       0.24964790132986496, 0.11450881474425720}};
  case 9:
    return {
      {-1.0000000000000000, -0.91073208942006030, -0.71126748591570886,
       -0.42635048571113896, -0.090373369606853298, 0.25613567083345540,
       0.57138304120873848, 0.81735278420041209, 0.96444016970527310},
      {0.024691358024691358, 0.14765401904631539, 0.24718937820459305,
       0.31684377567043798, 0.34827300277296659, 0.33769396697592959,
       0.28638669635723117, 0.20055329802455196, 0.090714504923282917}};
  default:
    throw std::runtime_error(
      "Only orders up to 9 supported for radau quadrature");
  }
  return {};
}

std::pair<std::vector<double>, std::vector<double>>
truncated_normal_rule(NormalRule rule)
{
  // from the "truncated normal quadrature" .python code
  switch (rule) {
  case NormalRule::SIGMA1:
    return {
      {0, 0.3436121352489559, 0.6473220334297102, 0.8706217027202311,
       0.9816974860670653},
      {0.2046394226558322 / 2, 0.1820146209511494, 0.128027596313765,
       0.06821017522834351, 0.01942789617882675}};
  case NormalRule::SIGMA2:
    return {
      {0, 0.2959590846054073, 0.5735693238435292, 0.8074757570903542,
       0.9607561326630086},
      {0.249758577881117 / 2, 0.2035976917174024, 0.1129523637830892,
       0.04552496709664563, 0.013045688462303995}};
  case NormalRule::SIGMA3:
    return {
      {0, 0.2661790968493327, 0.5263305051027921, 0.7664900509748058,
       0.9477581057921652},
      {0.3203929665957703 / 2, 0.2307493381206665, 0.08754316928625956,
       0.01882073900490792, 0.002690270290280566}};
  case NormalRule::HALFPOWER:
    return {
      {0, 0.315493297131259, 0.6016636608468, 0.8282105821126121,
       0.9662550592631028},
      {0.197723576944154 / 2, 0.1761766447490471, 0.1255723775152601,
       0.07163437433902098, 0.02775481492459504}};
  default: {
    throw std::runtime_error(
      "Only implemented 1-3, halfpower for truncated normal");
    return {};
  }
  }
}

std::pair<std::vector<vs::Vector>, std::vector<double>>
spherical_cap_radau(
  double gammav, int ntheta, int nphi, std::function<double(double)> wfunc)
{
  auto [xlocs, weights] = radau_rule(nphi);
  if (wfunc) {
    for (size_t j = 0; j < weights.size(); ++j) {
      weights[j] *= wfunc(xlocs[j]);
    }
  }
  return spherical_cap_quadrature(gammav, ntheta, xlocs, weights);
}

std::pair<std::vector<vs::Vector>, std::vector<double>>
spherical_cap_truncated_normal(double gammav, int ntheta, NormalRule rule)
{
  auto [xlocs, weights] = truncated_normal_rule(rule);
  // want the center of the truncated normal distribution at the pole of the
  // cap -> -1 . Weights are already for a [-1,1] range from the generator
  std::transform(xlocs.cbegin(), xlocs.cend(), xlocs.begin(), [](double x) {
    return 2 * x - 1;
  });
  // half range to start, then mapped back to [-1,1]
  std::transform(
    weights.cbegin(), weights.cend(), weights.begin(),
    [](double w) { return 4 * w; });

  return spherical_cap_quadrature(gammav, ntheta, xlocs, weights);
}

std::vector<vs::Vector>
make_radar_grid(double delta_phi, int nphi, int ntheta, vs::Vector axis)
{
  // probably a nicer way to do this but we're going to
  // create set of rays around a circle at different cone "phi" angles
  // by first creating the canonical case around (0,0,1) and rotating the
  // coordinate system so that (0,0,1) matches the axis

  axis.normalize();
  const auto canon_vector = vs::Vector(0, 0, 1);

  std::vector<vs::Vector> rays; // a cone grid oriented around (0,0,1)
  const auto transform = rotation_matrix(axis, canon_vector);
  // handle the geometric singularity to avoid putting n rays at zero
  rays.push_back(transform & canon_vector);

  for (int j = 1; j < nphi; ++j) {
    const double phi = (delta_phi / (nphi - 1)) * j;
    const auto zr = std::sin(phi);
    for (int i = 0; i < ntheta; ++i) {
      const double theta = (2 * M_PI / ntheta) * i;
      const auto xr = zr * std::cos(theta);
      const auto yr = zr * std::sin(theta);
      auto ray = transform & vs::Vector(xr, yr, 1);
      ray.normalize();
      rays.push_back(ray);
    }
  }
  return rays;
}
} // namespace details

void
LidarLOS::load(const YAML::Node& node, DataProbePostProcessing* probes)
{
  auto lidar_spec = node["lidar_specifications"];

  auto create_lidar = [&](const YAML::Node& node) {
    std::string output_type = "netcdf";
    get_if_present(node, "output", output_type, output_type);
    if (output_type == "dataprobes") {
      LidarLineOfSite lidarLOS;
      auto lidarDBSpec = lidarLOS.determine_line_of_site_info(node);
      ThrowRequireMsg(
        probes, "Lidar processing with dataprobe output "
                "requires valid data probe object");
      probes->add_external_data_probe_spec_info(lidarDBSpec.release());
    } else {
      if (node["radar_cone_grid"]) {
        const auto cone_grid_spec = node["radar_cone_grid"];
        if (cone_grid_spec.Type() != YAML::NodeType::Map) {
          throw std::runtime_error("must specify map for cone grid");
        }

        double phi = 0;
        get_required(cone_grid_spec, "cone_angle", phi);
        phi = convert::degrees_to_radians(phi);
        ThrowRequire(phi > 0);

        int nphi = 0;
        get_required(cone_grid_spec, "num_circles", nphi);
        nphi += 1; // don't count the center as a circle

        int ntheta = 0;
        get_required(cone_grid_spec, "lines_per_cone_circle", ntheta);

        const auto look_ahead_spec = node["radar_specifications"];
        if (!look_ahead_spec) {
          throw std::runtime_error(
            "Must specifiy radar specification for cone grid");
        }
        const auto base_axis = look_ahead_spec["axis"].as<Coordinates>();

        auto rays = details::make_radar_grid(
          phi, nphi, ntheta,
          vs::Vector(base_axis.x_, base_axis.y_, base_axis.z_));

        int j = 0;
        for (const auto& ray : rays) {
          lidars_.emplace_back();
          auto mod_node = YAML::Clone(node);
          mod_node["name"] =
            node["name"].as<std::string>() + "-grid-" + std::to_string(j++);

          mod_node["radar_specifications"]["axis"] =
            std::vector<double>{ray[0], ray[1], ray[2]};
          lidars_.back().load(mod_node);
        }
      } else if (node["radar_cone_filter"]) {
        radars_.emplace_back();
        radars_.back().load(node);
      } else {
        lidars_.emplace_back();
        lidars_.back().load(node);
      }
    }
  };

  if (lidar_spec) {
    const auto is_scalar = lidar_spec.Type() == YAML::NodeType::Map;
    if (is_scalar) {
      create_lidar(lidar_spec);
    } else {
      std::set<std::string> names;
      for (auto spec : lidar_spec) {
        if (!spec["name"]) {
          throw std::runtime_error("lidar sequence requires a name");
        }
        names.insert(spec["name"].as<std::string>());
        create_lidar(spec);
      }
      if (names.size() != lidar_spec.size()) {
        std::string msg = "Non unique file name for lidar detected: ";
        for (auto spec : lidar_spec) {
          msg += spec["name"].as<std::string>() + " ";
        }
        throw std::runtime_error(msg);
      }
    }
  }
} // namespace details

void
LidarLOS::output(
  const stk::mesh::BulkData& bulk,
  const stk::mesh::Selector& sel,
  const std::string& coords_name,
  double dt,
  double time)
{
  constexpr int max_output_per_step = 1000;
  for (auto& los : lidars_) {
    const double small = 1e-8 * dt;
    const double next_time = time + dt;
    int step_outputs = 0;
    while (los.time() < next_time - small &&
           step_outputs < max_output_per_step) {
      const double dtratio = (los.time() - time) / dt;
      los.output(bulk, sel, coords_name, dtratio);
      los.increment_time();
      ++step_outputs;
    }
    if (step_outputs == max_output_per_step) {
      NaluEnv::self().naluOutputP0()
        << "Warning: max lidar outputs, " << max_output_per_step
        << " per step reached";
    }
  }

  for (auto& los : radars_) {
    const double small = 1e-8 * dt;
    const double next_time = time + dt;
    int step_outputs = 0;
    while (los.time() < next_time - small &&
           step_outputs < max_output_per_step) {
      const double dtratio = (los.time() - time) / dt;
      los.output_cone_filtered(bulk, sel, coords_name, dtratio);
      los.increment_time();
      ++step_outputs;
    }
    if (step_outputs == max_output_per_step) {
      NaluEnv::self().naluOutputP0()
        << "Warning: max lidar outputs, " << max_output_per_step
        << " per step reached";
    }
  }
}

namespace details {
RadarFilter
parse_radar_filter(const YAML::Node& node)
{
  const auto filter_node = node["radar_cone_filter"];

  double gammav;
  get_required(filter_node, "cone_angle", gammav);
  gammav = convert::degrees_to_radians(gammav);

  std::string quad_type;
  get_required(filter_node, "quadrature_type", quad_type);

  int ntheta;
  get_required(filter_node, "lines_per_cone_circle", ntheta);

  const auto look_ahead_radar = node["radar_specifications"];
  if (!look_ahead_radar) {
    throw std::runtime_error("radar filtering must be used with radar");
  }

  if (quad_type == "radau") {
    int nphi = 9;
    get_if_present(filter_node, "radau_points", nphi, nphi);
    if (nphi < 1 || nphi > 9) {
      throw std::runtime_error("Only points 1 to 9 supported");
    }
    std::string weight_func = "unity";
    get_if_present(filter_node, "radau_weight_type", weight_func, weight_func);
    std::transform(
      weight_func.cbegin(), weight_func.cend(), weight_func.begin(), ::tolower);
    if (weight_func == "unity") {
      auto [rays, weights] = details::spherical_cap_radau(gammav, ntheta, nphi);
      return {rays, weights};
    } else if (weight_func == "gaussian_halfpower") {
      auto [rays, weights] =
        details::spherical_cap_radau(gammav, ntheta, nphi, [](double x) {
          return 1.234529105942581469654 * std::pow(2., -x * x);
        });
      return {rays, weights};
    } else {
      throw std::runtime_error("unrecognized weight function");
    }
  } else if (quad_type == "truncated_normal1") {
    auto [rays, weights] = details::spherical_cap_truncated_normal(
      gammav, ntheta, NormalRule::SIGMA1);
    return {rays, weights};
  } else if (quad_type == "truncated_normal2") {
    auto [rays, weights] = details::spherical_cap_truncated_normal(
      gammav, ntheta, NormalRule::SIGMA2);
    return {rays, weights};
  } else if (quad_type == "truncated_normal3") {
    auto [rays, weights] = details::spherical_cap_truncated_normal(
      gammav, ntheta, NormalRule::SIGMA3);
    return {rays, weights};
  } else if (quad_type == "truncated_normal_halfpower") {
    auto [rays, weights] = details::spherical_cap_truncated_normal(
      gammav, ntheta, NormalRule::HALFPOWER);
    return {rays, weights};
  } else {
    throw std::runtime_error(
      "invalid quadrature type, radau and truncated_normal{1,2,3} supported.");
    return {};
  }
}

} // namespace details
} // namespace nalu
} // namespace sierra
