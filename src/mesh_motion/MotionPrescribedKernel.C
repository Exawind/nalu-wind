
#include "mesh_motion/MotionPrescribedKernel.h"
#include "mesh_motion/NgpMotion.h"

#include <NaluEnv.h>
#include <NaluParsing.h>
#include <stdexcept>
#include <vector>
namespace sierra {
namespace nalu {

MotionPrescribedKernel::MotionPrescribedKernel(const YAML::Node& node)
  : NgpMotionKernel<MotionPrescribedKernel>()
{
  load(node);
}

void
MotionPrescribedKernel::load(const YAML::Node& node)
{
  // perturb start and end times with a small value for
  // accurate comparison with floats
  get_if_present(node, "start_time", startTime_, startTime_);
  startTime_ = startTime_ - DBL_EPSILON;

  get_if_present(node, "end_time", endTime_, endTime_);
  endTime_ = endTime_ + DBL_EPSILON;

  for (int d = 0; d < nalu_ngp::NDimMax; ++d) {
    origin_[d] = node["centroid"][d].as<double>();
  }

  std::vector<std::string> variables =
    node["variables"].as<std::vector<std::string>>();
  bool has_time = false;
  for (auto var : variables) {
    if (var == "time")
      has_time = true;
  }

  if (!has_time) {
    throw std::runtime_error(
      "Time information absent from a prescribed mesh motion block. Please "
      "append it.");
  }

  auto values = node["values"];

  std::vector<MotionValues> defined_motion_values;

  for (size_t irow = 0; irow < values.size(); ++irow) {
    auto row_values = values[irow];

    MotionValues new_values_to_add;

    for (size_t icol = 0; icol < row_values.size(); ++icol) {
      double new_value_to_store = row_values[icol].as<double>();
      if (variables[icol] == "time")
        new_values_to_add.time_value = new_value_to_store;
      else if (variables[icol] == "x_disp")
        new_values_to_add.x_disp[0] = new_value_to_store;
      else if (variables[icol] == "y_disp")
        new_values_to_add.x_disp[1] = new_value_to_store;
      else if (variables[icol] == "z_disp")
        new_values_to_add.x_disp[2] = new_value_to_store;
      else if (variables[icol] == "x_vel")
        new_values_to_add.vel[0] = new_value_to_store;
      else if (variables[icol] == "y_vel")
        new_values_to_add.vel[1] = new_value_to_store;
      else if (variables[icol] == "z_vel")
        new_values_to_add.vel[2] = new_value_to_store;
      else if (variables[icol] == "x_angular_vel")
        new_values_to_add.angular_vel[0] = new_value_to_store;
      else if (variables[icol] == "y_angular_vel")
        new_values_to_add.angular_vel[1] = new_value_to_store;
      else if (variables[icol] == "z_angular_vel")
        new_values_to_add.angular_vel[2] = new_value_to_store;
      else if (variables[icol] == "x_angular_disp")
        new_values_to_add.angular_disp[0] = new_value_to_store;
      else if (variables[icol] == "y_angular_disp")
        new_values_to_add.angular_disp[1] = new_value_to_store;
      else if (variables[icol] == "z_angular_disp")
        new_values_to_add.angular_disp[2] = new_value_to_store;
    }
    if (irow > 0) {
      if (
        new_values_to_add.time_value <
        defined_motion_values.back().time_value) {
        throw std::runtime_error(
          "Prescribed motion must be in chronological order.");
      }
    }
    defined_motion_values.push_back(new_values_to_add);
  }
  auto defined_values_host_ = Kokkos::View<double**, Kokkos::HostSpace>(
    "defined_motion_values_host", defined_motion_values.size(), 13);
  for (size_t irow = 0; irow < defined_motion_values.size(); ++irow) {
    defined_values_host_(irow, 0) = defined_motion_values[irow].time_value;

    defined_values_host_(irow, 1) = defined_motion_values[irow].x_disp[0];
    defined_values_host_(irow, 2) = defined_motion_values[irow].x_disp[1];
    defined_values_host_(irow, 3) = defined_motion_values[irow].x_disp[2];

    defined_values_host_(irow, 4) = defined_motion_values[irow].vel[0];
    defined_values_host_(irow, 5) = defined_motion_values[irow].vel[1];
    defined_values_host_(irow, 6) = defined_motion_values[irow].vel[2];

    defined_values_host_(irow, 7) = defined_motion_values[irow].angular_vel[0];
    defined_values_host_(irow, 8) = defined_motion_values[irow].angular_vel[1];
    defined_values_host_(irow, 9) = defined_motion_values[irow].angular_vel[2];

    defined_values_host_(irow, 10) =
      defined_motion_values[irow].angular_disp[0];
    defined_values_host_(irow, 11) =
      defined_motion_values[irow].angular_disp[1];
    defined_values_host_(irow, 12) =
      defined_motion_values[irow].angular_disp[2];
  }

  defined_motion_values_ = Kokkos::View<double**>(
    "defined_motion_values", defined_motion_values.size(), 13);

  Kokkos::deep_copy(defined_motion_values_, defined_values_host_);
}

KOKKOS_FUNCTION
mm::TransMatType
MotionPrescribedKernel::build_transformation(
  const double& time, const mm::ThreeDVecType& /* xyz */)
{
  mm::TransMatType transMat;

  if (time < (startTime_))
    return transMat;
  double motionTime = (time < endTime_) ? time : endTime_;

  // min_index corresponds to trajectory index that has the minimum difference
  // in time to current time in simulation
  int min_index = -1;
  for (int i = 0; i < defined_motion_values_.extent(0); ++i) {
    if (defined_motion_values_(i, 0) > motionTime) {
      min_index = i;
    }
  }

  if (min_index == -1)
    min_index = defined_motion_values_.extent(0) - 1;
  else
    min_index = Kokkos::min(min_index - 1, 0);

  auto relevant_motion =
    Kokkos::subview(defined_motion_values_, min_index, Kokkos::ALL());
  double current_displacement[6];
  for (int i = 0; i < 6; ++i)
    current_displacement[i] = 0.0;

  // Defer to specified displacements, else default back to velocities
  if (relevant_motion[1] < -1e15) {
    // Note indices here, [0] corresponds to time
    current_displacement[0] =
      relevant_motion[4] * (motionTime - relevant_motion[0]);
    current_displacement[1] =
      relevant_motion[5] * (motionTime - relevant_motion[0]);
    current_displacement[2] =
      relevant_motion[6] * (motionTime - relevant_motion[0]);
  } else {
    // Interp displacement if information is available. Assume lines between
    // motion points.
    if (min_index == (defined_motion_values_.extent(0) - 1)) {
      // Note indices here, [1-3] corresponds to x, y, and z displacements in
      // relevant_motion, result gets shifted over for mesh_motion API
      // calculations
      current_displacement[0] = relevant_motion[1];
      current_displacement[1] = relevant_motion[2];
      current_displacement[2] = relevant_motion[3];
    } else {
      auto next_motion =
        Kokkos::subview(defined_motion_values_, min_index + 1, Kokkos::ALL());
      double interp_ratio = (motionTime - relevant_motion[0]) /
                            (next_motion[0] - relevant_motion[0]);
      current_displacement[0] = (1.0 - interp_ratio) * relevant_motion[1] +
                                interp_ratio * next_motion[1];
      current_displacement[1] = (1.0 - interp_ratio) * relevant_motion[2] +
                                interp_ratio * next_motion[2];
      current_displacement[2] = (1.0 - interp_ratio) * relevant_motion[3] +
                                interp_ratio * next_motion[3];
    }
  }

  if (relevant_motion[10] < -1e15) {
    // Note indices here, [0] corresponds to time
    current_displacement[3] =
      relevant_motion[7] * (motionTime - relevant_motion[0]);
    current_displacement[4] =
      relevant_motion[8] * (motionTime - relevant_motion[0]);
    current_displacement[5] =
      relevant_motion[9] * (motionTime - relevant_motion[0]);
  } else {
    if (min_index == defined_motion_values_.extent(0) - 1) {
      // Note indices here, [10-13] corresponds to x, y, and z angular
      // displacements in relevant_motion, result gets shifted over for
      // mesh_motion API calculations
      current_displacement[3] = relevant_motion[10];
      current_displacement[4] = relevant_motion[11];
      current_displacement[5] = relevant_motion[12];
    } else {
      auto next_motion =
        Kokkos::subview(defined_motion_values_, min_index + 1, Kokkos::ALL());
      double interp_ratio = (motionTime - relevant_motion[0]) /
                            (next_motion[0] - relevant_motion[0]);
      current_displacement[3] = (1.0 - interp_ratio) * relevant_motion[10] +
                                interp_ratio * next_motion[10];
      current_displacement[4] = (1.0 - interp_ratio) * relevant_motion[11] +
                                interp_ratio * next_motion[11];
      current_displacement[5] = (1.0 - interp_ratio) * relevant_motion[12] +
                                interp_ratio * next_motion[12];
    }
  }

  // Build matrix for translating object to cartesian origin
  transMat[0 * mm::matSize + 3] = -origin_[0];
  transMat[1 * mm::matSize + 3] = -origin_[1];
  transMat[2 * mm::matSize + 3] = -origin_[2];

  const double rollX = current_displacement[3];
  const double pitchY = current_displacement[4];
  const double yawZ = current_displacement[5];

  const double cy = std::cos(0.5 * yawZ);
  const double sy = std::sin(0.5 * yawZ);
  const double cp = std::cos(0.5 * pitchY);
  const double sp = std::sin(0.5 * pitchY);
  const double cr = std::cos(0.5 * rollX);
  const double sr = std::sin(0.5 * rollX);

  double q0 = cr * cp * cy + sr * sp * sy;
  double q3 = sr * cp * cy - cr * sp * sy;
  double q1 = cr * sp * cy + sr * cp * sy;
  double q2 = cr * cp * sy - sr * sp * cy;

  const double n = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 /= n;
  q1 /= n;
  q2 /= n;
  q3 /= n;

  // rotation matrix based on quaternion
  mm::TransMatType tempMat;
  // 1st row
  tempMat[0 * mm::matSize + 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
  tempMat[0 * mm::matSize + 1] = 2.0 * (q1 * q2 - q0 * q3);
  tempMat[0 * mm::matSize + 2] = 2.0 * (q0 * q2 + q1 * q3);
  // 2nd row
  tempMat[1 * mm::matSize + 0] = 2.0 * (q1 * q2 + q0 * q3);
  tempMat[1 * mm::matSize + 1] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
  tempMat[1 * mm::matSize + 2] = 2.0 * (q2 * q3 - q0 * q1);
  // 3rd row
  tempMat[2 * mm::matSize + 0] = 2.0 * (q1 * q3 - q0 * q2);
  tempMat[2 * mm::matSize + 1] = 2.0 * (q0 * q1 + q2 * q3);
  tempMat[2 * mm::matSize + 2] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

  // composite addition of motions in current group
  transMat = add_motion(tempMat, transMat);

  // Build matrix for translating object back to its origin
  tempMat = mm::TransMatType::I();
  tempMat[0 * mm::matSize + 3] = origin_[0] + current_displacement[0];
  tempMat[1 * mm::matSize + 3] = origin_[1] + current_displacement[1];
  tempMat[2 * mm::matSize + 3] = origin_[2] + current_displacement[2];

  // composite addition of motions
  return add_motion(tempMat, transMat);
}

KOKKOS_FUNCTION
mm::ThreeDVecType
MotionPrescribedKernel::compute_velocity(
  const double& time,
  const mm::TransMatType& compTrans,
  const mm::ThreeDVecType& /* mxyz */,
  const mm::ThreeDVecType& cxyz)
{
  mm::ThreeDVecType vel;

  if ((time < startTime_) || (time > endTime_))
    return vel;

  double motionTime = (time < endTime_) ? time : endTime_;
  // min_index corresponds to trajectory index that has the minimum difference
  // in time to current time in simulation
  int min_index = -1;
  for (int i = 0; i < defined_motion_values_.extent(0); ++i) {
    if (defined_motion_values_(i, 0) > motionTime) {
      min_index = i;
    }
  }

  if (min_index == -1)
    min_index = defined_motion_values_.extent(0) - 1;
  else
    min_index = Kokkos::min(min_index - 1, 0);

  auto relevant_motion =
    Kokkos::subview(defined_motion_values_, min_index, Kokkos::ALL());

  auto motion_np1 =
    (min_index == (defined_motion_values_.extent(0) - 1))
      ? relevant_motion
      : Kokkos::subview(defined_motion_values_, min_index + 1, Kokkos::ALL());
  double time_between = motion_np1[0] - relevant_motion[0];

  mm::ThreeDVecType transOrigin;
  for (int d = 0; d < nalu_ngp::NDimMax; d++) {
    transOrigin[d] = compTrans[d * mm::matSize + 0] * origin_[0] +
                     compTrans[d * mm::matSize + 1] * origin_[1] +
                     compTrans[d * mm::matSize + 2] * origin_[2] +
                     compTrans[d * mm::matSize + 3];
  }
  mm::ThreeDVecType relCoord;
  mm::ThreeDVecType vecOmega;
  for (int d = 0; d < nalu_ngp::NDimMax && relevant_motion[1] > -1e15; d++) {
    // Note the indices here, [7, 8, 9] correspond to angular velocities
    relCoord[d] = cxyz[d] - transOrigin[d];
    vecOmega[d] = relevant_motion[7 + d];
  }

  for (int d = 0; d < nalu_ngp::NDimMax && relevant_motion[1] < -1e15 &&
                  time_between > 1e-12;
       d++) {
    // Note the indices here, [10, 11, 12] correspond to angular displacements
    relCoord[d] = cxyz[d] - transOrigin[d];
    vecOmega[d] = (motion_np1[10 + d] - relevant_motion[10 + d]) / time_between;
  }

  vel[0] = vecOmega[1] * relCoord[2] - vecOmega[2] * relCoord[1];
  vel[1] = vecOmega[2] * relCoord[0] - vecOmega[0] * relCoord[2];
  vel[2] = vecOmega[0] * relCoord[1] - vecOmega[1] * relCoord[0];

  mm::ThreeDVecType vecVel;
  for (int d = 0; d < nalu_ngp::NDimMax && relevant_motion[1] < -1e15; d++) {
    // Note the indices here, [4, 5, 6] correspond to translational velocities
    vecVel[d] = relevant_motion[4 + d];
  }

  for (int d = 0; d < nalu_ngp::NDimMax && relevant_motion[1] > -1e15 &&
                  time_between > 1e-12;
       d++) {
    // Note the indices here, [1, 2, 3] correspond to translational
    // displacements
    vecVel[d] = (motion_np1[1 + d] - relevant_motion[1 + d]) / time_between;
  }

  vel[0] += vecVel[0];
  vel[1] += vecVel[1];
  vel[2] += vecVel[2];

  return vel;
}

} // namespace nalu
} // namespace sierra
