
#include <wind_energy/SyntheticLidar.h>

#include <NaluParsing.h>
#include <nalu_make_unique.h>
#include <master_element/TensorOps.h>

#include <xfer/Transfer.h>

namespace sierra {
namespace nalu {

SpinnerLidarSegmentGenerator::SpinnerLidarSegmentGenerator(
  PrismParameters inner,
  PrismParameters outer,
  double in_beamLength)
: innerPrism_(inner),
  outerPrism_(outer),
  beamLength_(in_beamLength)
{}

void
SpinnerLidarSegmentGenerator::load(const YAML::Node& node)
{
  NaluEnv::self().naluOutputP0() << "LidarLineOfSite::SpinnerLidarSegmentGenerator::load" << std::endl;

  ThrowRequireMsg(node["center"], "Lidar center must be provided");
  set_lidar_center(node["center"].as<Coordinates>());

  ThrowRequireMsg(node["axis"], "Lidar axis must be provided");
  set_laser_axis(node["axis"].as<Coordinates>());

  double innerPrismTheta0 = 90;
  get_if_present(node, "inner_prism_initial_theta", innerPrismTheta0, innerPrismTheta0);
  innerPrismTheta0 *= M_PI / 180;

  double innerPrismRot = 3.5;
  get_if_present(node, "inner_prism_rotation_rate", innerPrismRot);
  innerPrismRot *= 2 * M_PI;

  double innerPrismAzi = 15.2;
  get_if_present(node, "inner_prism_azimuth", innerPrismAzi);
  innerPrismAzi *= M_PI / 180;

  set_inner_prism(innerPrismTheta0, innerPrismRot, innerPrismAzi);

  double outerPrismTheta0 = 90;
  get_if_present(node, "outer_prism_initial_theta", outerPrismTheta0);
  outerPrismTheta0 *= M_PI / 180;

  double outerPrismRot = 6.5;
  get_if_present(node, "outer_prism_rotation_rate", outerPrismRot);
  outerPrismRot *= 2 * M_PI;

  double outerPrismAzi = 15.2;
  get_if_present(node, "outer_prism_azimuth", outerPrismAzi);
  outerPrismAzi *= M_PI / 180;

  set_outer_prism(outerPrismTheta0, outerPrismRot, outerPrismAzi);

  double beamLength = 1;
  get_required(node, "beam_length", beamLength);
  set_beam_length(beamLength);

  if (node["ground_direction"]) {
    set_ground_normal(node["ground_direction"].as<Coordinates>());
  }

  ThrowRequireMsg(std::abs(ddot(groundNormal_.data(), laserAxis_.data(),3)) < small_positive_value(),
    "Ground and laser axes must be orthogonal");
}

namespace {
std::array<double,3>
rotate_euler_vec(const std::array<double, 3>& axis, double angle, std::array<double, 3> vec)
{
  enum {XH = 0, YH = 1, ZH = 2};

  normalize_vec3(vec.data());

  std::array<double, 9> nX = {{
      0, -axis[ZH], +axis[YH],
      +axis[ZH], 0, -axis[XH],
      -axis[YH], +axis[XH], 0
  }};
  const double cosTheta = std::cos(angle);

  std::array<double, 9> rot = {{
      cosTheta, 0, 0,
      0, cosTheta, 0,
      0, 0, cosTheta
  }};

  const double sinTheta = std::sin(angle);
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      rot[j * 3 + i] += (1 - cosTheta) * axis[i] * axis[j] + sinTheta * nX[j * 3 + i];
    }
  }

  std::array<double, 3> vecprime;
  matvec33(rot.data(), vec.data(), vecprime.data());
  return vecprime;
}

std::array<double, 3>
reflect(const std::array<double, 3>& line, const std::array<double ,3>& vec)
{
  enum {XH = 0, YH = 1, ZH = 2};

  std::array<double, 9> ref = {{
      1 - 2 * line[XH] * line[XH],   - 2 * line[XH] * line[YH],   - 2 * line[XH] * line[ZH],
        - 2 * line[YH] * line[XH], 1 - 2 * line[YH] * line[YH],   - 2 * line[YH] * line[ZH],
        - 2 * line[ZH] * line[XH],   - 2 * line[ZH] * line[YH], 1 - 2 * line[ZH] * line[ZH]
  }};

  std::array<double, 3> result;
  matvec33(ref.data(), vec.data(), result.data());
  return result;
}
}

Segment
SpinnerLidarSegmentGenerator::generate_path_segment(double time) const
{
  auto axis = laserAxis_;
  normalize_vec3(axis.data());

  const double innerTheta = innerPrism_.theta0_ + innerPrism_.rot_ * time;
  const double outerTheta = outerPrism_.theta0_ + outerPrism_.rot_ * time;

  const auto reflection_1 = rotate_euler_vec(
    axis,
    innerTheta,
    rotate_euler_vec(groundNormal_, -(innerPrism_.azimuth_/2 + M_PI/2), axis )
  );

  const auto reflection_2 = rotate_euler_vec(
    axis,
    outerTheta,
    rotate_euler_vec(groundNormal_, outerPrism_.azimuth_/2, axis)
  );

  Segment current;
  current.tail_ =  lidarCenter_;

  std::array<double,3 > reversedAxis = {{-axis[0], -axis[1], -axis[2]}};
  current.tip_ = reflect(reflection_2, reflect(reflection_1, reversedAxis));

  for (int d = 0; d < 3; ++d) {
    current.tip_[d] = current.tail_[d] + current.tip_[d] * beamLength_;
  }

  return current;
}

void
LidarLineOfSite::load(const YAML::Node& node)
{
  NaluEnv::self().naluOutputP0() << "LidarLineOfSite::load" << std::endl;
  get_required(node, "scan_time", scanTime_);
  get_required(node, "number_of_samples", nsamples_);
  get_required(node, "points_along_line", npoints_);

  if (node["name"]) {
    name_ = node["name"].as<std::string>();
  }

  const YAML::Node fromTargets = node["from_target_part"];
  if (fromTargets.Type() == YAML::NodeType::Scalar) {
    fromTargetNames_.push_back(fromTargets.as<std::string>());
  }
  else {
    for (const auto& target : fromTargets){
      fromTargetNames_.push_back(target.as<std::string>());
    }
  }
  segGen.load(node);
}


std::unique_ptr<DataProbeSpecInfo>
LidarLineOfSite::determine_line_of_site_info(const YAML::Node& node)
{
  load(node);

  auto lidarLOSInfo = make_unique<DataProbeSpecInfo>();

  lidarLOSInfo->xferName_ = "LidarSampling_xfer";
  lidarLOSInfo->fromToName_.emplace_back("velocity", "velocity_probe");
  lidarLOSInfo->fieldInfo_.emplace_back("velocity_probe", 3);
  lidarLOSInfo->fromTargetNames_ = fromTargetNames_;

  auto probeInfo = make_unique<DataProbeInfo>();

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

  const int numProcs = NaluEnv::self().parallel_size();
  const int divProcProbe = std::max(numProcs/nsamples_, numProcs);

  for (int ilos = 0; ilos < nsamples_; ilos++) {
    const double lidarTime = scanTime_ / (double)nsamples_ * ilos;
    Segment seg = segGen.generate_path_segment(lidarTime);

    probeInfo->processorId_[ilos] = divProcProbe > 0 ? ilos % divProcProbe : 0;
    probeInfo->partName_[ilos] = name_ + "_" + std::to_string(ilos);
    probeInfo->numPoints_[ilos] = npoints_;

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


} // namespace nalu
} // namespace sierra
