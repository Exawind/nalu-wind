#include "aero/fmb/KynemaFMBBase.h"

#include <stk_mesh/base/Selector.hpp>

namespace sierra {

namespace kynema_ugf {

void
KynemaFMBBase::map_displacements_point(PointMass& point, bool updateCur)
{
  auto& meta = point.bulk->mesh_meta_data();
  const VectorFieldType* modelCoords =
    meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
  VectorFieldType* curCoords =
    meta.get_field<double>(stk::topology::NODE_RANK, "current_coordinates");
  VectorFieldType* displacement =
    meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");

  VectorFieldType* meshVelocity =
    meta.get_field<double>(stk::topology::NODE_RANK, "mesh_velocity");

  modelCoords->sync_to_host();
  curCoords->sync_to_host();
  displacement->sync_to_host();
  meshVelocity->sync_to_host();

  std::array<double, 7> translation_and_rotation_position = point.p_data.pos;
  std::array<double, 6> translation_and_rotation_velocities = point.p_data.vel;
  std::array<double, 3> current_center_of_mass_location = {
    translation_and_rotation_position[0], translation_and_rotation_position[1],
    translation_and_rotation_position[2]};

  auto q0 = translation_and_rotation_position[3];
  auto q1 = translation_and_rotation_position[4];
  auto q2 = translation_and_rotation_position[5];
  auto q3 = translation_and_rotation_position[6];

  std::array<std::array<double, 3>, 3> current_rotation_matrix = {
    {{q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3),
      2.0 * (q0 * q2 + q1 * q3)},
     {2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
      2.0 * (q2 * q3 - q0 * q1)},
     {2.0 * (q1 * q3 - q0 * q2), 2.0 * (q0 * q1 + q2 * q3),
      q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3}}};

  std::array<double, 3> new_point = {0.0, 0.0, 0.0};
  std::array<double, 3> current_point = {0.0, 0.0, 0.0};
  std::array<double, 3> new_velocity = {0.0, 0.0, 0.0};
  std::array<double, 3> lever_arm = {0.0, 0.0, 0.0};

  stk::mesh::Selector sel(stk::mesh::selectUnion(point.moving_mesh_blocks));
  const auto& bkts = point.bulk->get_buckets(stk::topology::NODE_RANK, sel);

  auto cross_product = [](double* a, double* b, double* axb) {
    axb[0] = a[1] * b[2] - a[2] * b[1];
    axb[1] = a[2] * b[0] - a[0] * b[2];
    axb[2] = a[0] * b[1] - a[1] * b[0];
  };

  for (auto b : bkts) {
    for (size_t in = 0; in < b->size(); in++) {
      auto node = (*b)[in];
      double* modelc = stk::mesh::field_data(*modelCoords, node);
      double* disp = stk::mesh::field_data(*displacement, node);
      double* currc = stk::mesh::field_data(*curCoords, node);
      double* meshv = stk::mesh::field_data(*meshVelocity, node);

      for (int row = 0; row < 3; ++row) {
        current_point[row] = modelc[row] - point.p_data.center_of_mass[row];
      }

      for (int row = 0; row < 3; ++row) {
        new_point[row] = 0.0;
        for (int col = 0; col < 3; ++col) {
          new_point[row] +=
            current_rotation_matrix[row][col] * current_point[col];
        }
        new_point[row] += current_center_of_mass_location[row];
      }

      for (int row = 0; row < 3; ++row) {
        disp[row] = new_point[row] - modelc[row];
        lever_arm[row] = new_point[row] - current_center_of_mass_location[row];
      }

      cross_product(
        &translation_and_rotation_velocities[3], lever_arm.data(),
        new_velocity.data());

      for (int row = 0; row < 3; ++row) {
        new_velocity[row] += translation_and_rotation_velocities[row];
        meshv[row] = new_velocity[row];
      }

      if (updateCur) {
        for (int row = 0; row < 3; ++row) {
          currc[row] = new_point[row];
        }
      }
    }
  }

  // Note this syncs too much as is. Ideally above is done on device.
  curCoords->modify_on_host();
  displacement->modify_on_host();
  meshVelocity->modify_on_host();
  curCoords->sync_to_device();
  displacement->sync_to_device();
  meshVelocity->sync_to_device();
}

void
KynemaFMBBase::map_loads_point(PointMass& point)
{
  point.calc_loads->initialize();
  point.calc_loads->execute();

  auto& meta = point.bulk->mesh_meta_data();
  const VectorFieldType* modelCoords =
    meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
  const VectorFieldType* meshDisp =
    meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");

  std::array<double, 7> translation_and_rotation_position = point.p_data.pos;

  std::array<double, 3> center_of_mass = {
    translation_and_rotation_position[0], translation_and_rotation_position[1],
    translation_and_rotation_position[2]};

  auto forces_and_moments = fsi::accumulateLoadsAndMoments(
    *(point.bulk), point.forcing_surfaces, *modelCoords, *meshDisp,
    *(point.total_force), center_of_mass);

  // Reduce to get full result and then feed into open turbine
  MPI_Allreduce(
    MPI_IN_PLACE, forces_and_moments.data(), 6, MPI_DOUBLE, MPI_SUM,
    point.bulk->parallel());

  point.p_data.loads = forces_and_moments;
}

} // namespace kynema_ugf
} // namespace sierra
