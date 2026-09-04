#include "aero/fmb/KynemaFMBBase.h"
#include "aero/fmb/KynemaFMBBeamUtils.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include "stk_mesh/base/GetNgpMesh.hpp"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpTypes.h"

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

void
KynemaFMBBase::compute_mapping_beam(BeamBody& beam)
{

  Kokkos::deep_copy(beam.beam_data.node_xi, beam.beam_data_host.node_xi);
  Kokkos::deep_copy(beam.beam_data.pos, beam.beam_data_host.pos);

  constexpr int kMaxInterpolationNodes = 32;
  auto node_xi_host = beam.beam_data_host.node_xi;
  auto n_nodes = beam.n_nodes;
  // Precompute barycentric weights for interpolation on xi-space nodes
  ComputeBarycentricWeights(
    node_xi_host.data(), static_cast<int>(n_nodes),
    beam.beam_data_host.bary_weights.data());
  Kokkos::deep_copy(
    beam.beam_data.bary_weights, beam.beam_data_host.bary_weights);

  auto& meta = beam.bulk->mesh_meta_data();
  const auto& ngpMesh = stk::mesh::get_updated_ngp_mesh(*(beam.bulk));
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::selectUnion(beam.moving_mesh_blocks) &
    (meta.locally_owned_part() | meta.globally_shared_part());
  // get the field from the NGP mesh
  stk::mesh::NgpField<double> modelCoords =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "coordinates"));
  stk::mesh::NgpField<double> beam_xi =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "beam_xi"));
  modelCoords.sync_to_device();
  beam_xi.sync_to_device();

  auto positions = beam.beam_data.pos;
  auto node_xi = beam.beam_data.node_xi;
  auto bary_weights = beam.beam_data.bary_weights;

  kynema_ugf_ngp::run_entity_algorithm(
    "KynemaFMBBeam_compute_mapping", ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(
      const kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {
      const double query_point[3] = {
        modelCoords(mi, 0), modelCoords(mi, 1), modelCoords(mi, 2)};
      double scratch_weights[kMaxInterpolationNodes];
      double scratch_dweights[kMaxInterpolationNodes];
      double closest_position[3] = {0.0, 0.0, 0.0};
      double xi = 0.0;
      double dist2 = 0.0;

      FindClosestPointOnBlade(
        query_point, node_xi.data(), positions.data(), n_nodes,
        bary_weights.data(), scratch_weights, scratch_dweights, xi,
        closest_position, dist2);

      beam_xi.get(mi, 0) = xi;
    });

  beam_xi.modify_on_device();
}

void
KynemaFMBBase::map_displacements_beam(BeamBody& beam, bool updateCur)
{

  Kokkos::deep_copy(beam.beam_data.disp, beam.beam_data_host.disp);
  Kokkos::deep_copy(beam.beam_data.vel, beam.beam_data_host.vel);

  constexpr int kMaxInterpolationNodes = 32;

  auto& meta = beam.bulk->mesh_meta_data();
  const auto& ngpMesh = stk::mesh::get_updated_ngp_mesh(*(beam.bulk));
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::selectUnion(beam.moving_mesh_blocks) &
    (meta.locally_owned_part() | meta.globally_shared_part());
  // get the field from the NGP mesh
  stk::mesh::NgpField<double> modelCoords =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "coordinates"));
  stk::mesh::NgpField<double> curCoords =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "current_coordinates"));
  stk::mesh::NgpField<double> meshDisp =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "mesh_displacement"));
  stk::mesh::NgpField<double> meshVel =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "mesh_velocity"));
  stk::mesh::NgpField<double> beam_xi =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "beam_xi"));

  modelCoords.sync_to_device();
  curCoords.sync_to_device();
  meshDisp.sync_to_device();
  meshVel.sync_to_device();
  beam_xi.sync_to_device();

  auto n_nodes = beam.n_nodes;

  auto displacements = beam.beam_data.disp;
  auto velocities = beam.beam_data.vel;
  auto node_xi = beam.beam_data.node_xi;
  auto positions = beam.beam_data.pos;
  auto bary_weights = beam.beam_data.bary_weights;

  kynema_ugf_ngp::run_entity_algorithm(
    "KynemaFMBBeam_map_displacements", ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(
      const kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {
      double scratch_weights[kMaxInterpolationNodes];
      double pos[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double disp[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double vel[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      const double xi = beam_xi.get(mi, 0);
      double qp[3] = {
        modelCoords(mi, 0), modelCoords(mi, 1), modelCoords(mi, 2)};

      InterpolateFieldAtPoint(
        xi, node_xi.data(), positions.data(), n_nodes, 7, bary_weights.data(),
        scratch_weights, pos);
      InterpolateFieldAtPoint(
        xi, node_xi.data(), displacements.data(), n_nodes, 7,
        bary_weights.data(), scratch_weights, disp);
      InterpolateFieldAtPoint(
        xi, node_xi.data(), velocities.data(), n_nodes, 6, bary_weights.data(),
        scratch_weights, vel);

      double rel_pos_g[3] = {qp[0] - pos[0], qp[1] - pos[1], qp[2] - pos[2]};
      double rel_pos_l[3] = {0.0, 0.0, 0.0};
      RotateVectorByQuaternionInv(pos + 3, rel_pos_g, rel_pos_l);

      double tmp_disp[3] = {0.0, 0.0, 0.0};
      double rot_disp[3] = {0.0, 0.0, 0.0};
      RotateVectorByQuaternion(disp + 3, rel_pos_l, tmp_disp);
      RotateVectorByQuaternion(pos + 3, tmp_disp, rot_disp);

      meshDisp.get(mi, 0) = disp[0] + rot_disp[0] - rel_pos_g[0];
      meshDisp.get(mi, 1) = disp[1] + rot_disp[1] - rel_pos_g[1];
      meshDisp.get(mi, 2) = disp[2] + rot_disp[2] - rel_pos_g[2];

      const double omega_x = vel[3];
      const double omega_y = vel[4];
      const double omega_z = vel[5];
      const double rot_vx = omega_y * rot_disp[2] - omega_z * rot_disp[1];
      const double rot_vy = omega_z * rot_disp[0] - omega_x * rot_disp[2];
      const double rot_vz = omega_x * rot_disp[1] - omega_y * rot_disp[0];
      meshVel.get(mi, 0) = vel[0] + rot_vx;
      meshVel.get(mi, 1) = vel[1] + rot_vy;
      meshVel.get(mi, 2) = vel[2] + rot_vz;

      if (updateCur) {
        curCoords.get(mi, 0) = modelCoords(mi, 0) + meshDisp.get(mi, 0);
        curCoords.get(mi, 1) = modelCoords(mi, 1) + meshDisp.get(mi, 1);
        curCoords.get(mi, 2) = modelCoords(mi, 2) + meshDisp.get(mi, 2);
      }
    });

  meshDisp.modify_on_device();
  meshVel.modify_on_device();
  curCoords.modify_on_device();
}

void
KynemaFMBBase::map_loads_beam(BeamBody& beam)
{
  // First calculate the assembled forces on the beam surfaces
  beam.calc_loads->initialize();
  beam.calc_loads->execute();

  auto& meta = beam.bulk->mesh_meta_data();
  const auto& ngpMesh = stk::mesh::get_updated_ngp_mesh(*(beam.bulk));
  const stk::mesh::EntityRank entityRank = stk::topology::NODE_RANK;

  // get the parts in the current motion frame
  stk::mesh::Selector sel =
    stk::mesh::selectUnion(beam.forcing_surfaces) &
    (meta.locally_owned_part() | meta.globally_shared_part());
  stk::mesh::NgpField<double> modelCoords =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "coordinates"));
  stk::mesh::NgpField<double> tforce = stk::mesh::get_updated_ngp_field<double>(
    *meta.get_field<double>(entityRank, "tforce"));
  stk::mesh::NgpField<double> beam_xi =
    stk::mesh::get_updated_ngp_field<double>(
      *meta.get_field<double>(entityRank, "beam_xi"));

  tforce.sync_to_device();
  beam_xi.sync_to_device();

  constexpr int kMaxInterpolationNodes = 32;
  auto n_nodes = beam.n_nodes;
  auto bary_weights = beam.beam_data.bary_weights;
  auto node_xi = beam.beam_data.node_xi;
  auto positions = beam.beam_data.pos;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      beam.beam_data_host.loads(i, j) = 0.0;
    }
  }
  Kokkos::deep_copy(beam.beam_data.loads, beam.beam_data_host.loads);
  auto loads = beam.beam_data.loads;

  kynema_ugf_ngp::run_entity_algorithm(
    "KynemaFMBBeam_map_loads", ngpMesh, entityRank, sel,
    KOKKOS_LAMBDA(
      const kynema_ugf_ngp::NGPMeshTraits<stk::mesh::NgpMesh>::MeshIndex& mi) {
      const double query_point[3] = {
        modelCoords(mi, 0), modelCoords(mi, 1), modelCoords(mi, 2)};
      double scratch_weights[kMaxInterpolationNodes];
      double closest_position[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      double xi = beam_xi.get(mi, 0);
      const double force[3] = {tforce(mi, 0), tforce(mi, 1), tforce(mi, 2)};
      double moment[3] = {0.0, 0.0, 0.0};

      // Get the closest position on the beam to the query point and compute the
      // relative position vector Relative position vector is expected to not
      // change with beam deformation
      InterpolateFieldAtPoint(
        xi, node_xi.data(), positions.data(), n_nodes, 7, bary_weights.data(),
        scratch_weights, closest_position);

      double rel_pos[3] = {
        query_point[0] - closest_position[0],
        query_point[1] - closest_position[1],
        query_point[2] - closest_position[2]};

      // Transferring force from beam surface to beam node will add moment
      CrossProduct3(rel_pos, force, moment);

      // Compute all shape functions for the current xi value to distribute the
      // force and moment to all beam nodes
      LagrangePolynomialInterpWeights(
        xi, node_xi.data(), bary_weights.data(), n_nodes, scratch_weights);

      // Multiply the force and moment by the shape function weights and add to
      // the loads on each beam node
      for (std::size_t i = 0; i < n_nodes; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
          Kokkos::atomic_add(&loads(i, j), scratch_weights[i] * force[j]);
          Kokkos::atomic_add(&loads(i, j + 3), scratch_weights[i] * moment[j]);
        }
      }
    });

  Kokkos::fence();
  Kokkos::deep_copy(beam.beam_data_host.loads, beam.beam_data.loads);

  // Sum the loads across all ranks to get the total load on each beam node
  MPI_Allreduce(
    MPI_IN_PLACE, beam.beam_data_host.loads.data(), 6 * beam.n_nodes,
    MPI_DOUBLE, MPI_SUM, beam.bulk->parallel());
}

} // namespace kynema_ugf
} // namespace sierra
