// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "KynemaUGFEnv.h"
#include "aero/fmb/KynemaFMBSixDof.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include <fstream>
#include <KynemaUGFParsing.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace sierra {

namespace kynema_ugf {

KynemaFMBSixDof::KynemaFMBSixDof(const YAML::Node& node)
  : enable_calc_loads_(true)
{
  load(node);
}

void
KynemaFMBSixDof::load_point(const YAML::Node& node)
{

  PointMassInterface new_iface;
  PointMass new_body;
  const int ndim = 3;
  const int tensor_ndim = ndim * ndim;

  assert(node["moments_of_inertia"]);
  assert(node["moments_of_inertia"].size() == tensor_ndim);
  assert(node["center_of_mass"]);
  assert(node["center_of_mass"].size() == ndim);
  assert(node["mass"]);

  if (node["output_file_name"])
    new_iface.output_file_name = node["output_file_name"].as<std::string>();

  if (node["use_restart_data"])
    new_iface.use_restart_data = node["use_restart_data"].as<bool>();

  if (node["number_of_nonlinear_iterations"])
    new_iface.number_of_nonlinear_iterations =
      node["number_of_nonlinear_iterations"].as<int>();

  if (node["damping_factor"])
    new_iface.rho_inf = node["damping_factor"].as<double>();

  for (int d = 0; d < tensor_ndim; ++d) {
    new_iface.moments_of_inertia[d] =
      node["moments_of_inertia"][d].as<double>();
  }
  for (int d = 0; d < ndim; ++d) {
    new_iface.center_of_mass[d] = node["center_of_mass"][d].as<double>();
    new_body.p_data.center_of_mass[d] = node["center_of_mass"][d].as<double>();
  }
  if (node["initial_displacement"]) {
    for (int d = 0; d < ndim; ++d) {
      new_iface.disp_init[d] = node["initial_displacement"][d].as<double>();
    }
  }
  std::array<double, 3> theta_init = {0.0, 0.0, 0.0};
  if (node["initial_rotational_displacement"]) {
    for (int d = 0; d < ndim; ++d) {
      theta_init[d] = node["initial_rotational_displacement"][d].as<double>();
    }
  }
  // Convert to quaternions (ZYX order)
  const double c1 = cos(theta_init[2] / 2.);
  const double s1 = sin(theta_init[2] / 2.);
  const double c2 = cos(theta_init[1] / 2.);
  const double s2 = sin(theta_init[1] / 2.);
  const double c3 = cos(theta_init[0] / 2.);
  const double s3 = sin(theta_init[0] / 2.);
  new_iface.q_init[0] = c1 * c2 * c3 - s1 * s2 * s3;
  new_iface.q_init[1] = c1 * c2 * s3 - s1 * s2 * c3;
  new_iface.q_init[2] = c1 * s2 * c3 - s1 * c2 * s3;
  new_iface.q_init[3] = s1 * c2 * c3 - c1 * s2 * s3;
  if (node["initial_velocity"]) {
    for (int d = 0; d < ndim; ++d) {
      new_iface.v_init[d] = node["initial_velocity"][d].as<double>();
    }
  }
  if (node["initial_rotational_velocity"]) {
    for (int d = 0; d < ndim; ++d) {
      new_iface.omega_init[d] =
        node["initial_rotational_velocity"][d].as<double>();
    }
  }
  if (node["initial_acceleration"]) {
    for (int d = 0; d < ndim; ++d) {
      new_iface.a_init[d] = node["initial_acceleration"][d].as<double>();
    }
  }
  if (node["initial_rotational_acceleration"]) {
    for (int d = 0; d < ndim; ++d) {
      new_iface.alpha_init[d] =
        node["initial_rotational_acceleration"][d].as<double>();
    }
  }

  new_iface.mass = node["mass"].as<double>();

  if (node["forcing_surfaces"]) {
    for (std::size_t isurf = 0; isurf < node["forcing_surfaces"].size();
         ++isurf) {
      new_body.forcing_surface_names.emplace_back(
        node["forcing_surfaces"][isurf].as<std::string>());
    }
  }

  if (node["moving_mesh_blocks"]) {
    for (std::size_t iblock = 0; iblock < node["moving_mesh_blocks"].size();
         ++iblock) {
      new_body.moving_mesh_block_names.emplace_back(
        node["moving_mesh_blocks"][iblock].as<std::string>());
    }
  }

  if (
    node["tethers_initial_length"] || node["tethers_stiffness"] ||
    node["tethers_fairlead_position"] || node["tethers_anchor_position"]) {

    assert(node["tethers_initial_length"]);
    assert(node["tethers_stiffness"]);
    assert(node["tethers_fairlead_position"]);
    assert(node["tethers_anchor_position"]);

    assert(
      node["tethers_initial_length"].size() ==
      node["tethers_stiffness"].size());
    assert(
      ndim * node["tethers_initial_length"].size() ==
      node["tethers_fairlead_position"].size());
    assert(
      ndim * node["tethers_initial_length"].size() ==
      node["tethers_anchor_position"].size());

    const int number_of_tethers = node["tethers_initial_length"].size();

    for (int itether = 0; itether < number_of_tethers; ++itether) {

      new_iface.tethers.emplace_back(Tether());
      auto&& tether = new_iface.tethers.back();

      tether.stiffness = node["tethers_stiffness"][itether].as<double>();
      tether.initial_length =
        node["tethers_initial_length"][itether].as<double>();
      for (int d = 0; d < ndim; ++d) {
        tether.fairlead_position[d] =
          node["tethers_fairlead_position"][3 * itether + d].as<double>();
        tether.anchor_position[d] =
          node["tethers_anchor_position"][3 * itether + d].as<double>();
      }
    }
  }
  point_bodies_.emplace_back(new_body);
  point_interfaces_.emplace_back(new_iface);
}

void
KynemaFMBSixDof::load(const YAML::Node& node)
{
  const int ndim = 3;
  get_required(node, "number_of_bodies", number_of_bodies_);

  if (node["gravity"]) {
    for (int idim = 0; idim < ndim; ++idim) {
      gravity_[idim] = node["gravity"][idim].as<double>();
    }
  }

  for (int ibody = 0; ibody < number_of_bodies_; ++ibody) {
    if (!node["Body" + std::to_string(ibody)]) {
      throw std::runtime_error(
        "Node for Body" + std::to_string(ibody) +
        "not present or correct in input file");
    }

    auto body_node = node["Body" + std::to_string(ibody)];

    assert(body_node["type"]);

    std::string body_type = body_node["type"].as<std::string>();

    if (body_type == "point") {
      load_point(body_node);
    } else {
      throw std::runtime_error(
        "unrecognized body type for 6 DOF. Currently only point is supported.");
    }
  }
}

void
KynemaFMBSixDof::setup_point(
  PointMass& point,
  PointMassInterface& iface,
  const double dtKynemaUGF,
  std::shared_ptr<stk::mesh::BulkData> bulk)
{

  auto mass_matrix = std::array{
    std::array{iface.mass, 0., 0., 0., 0., 0.},
    std::array{0., iface.mass, 0., 0., 0., 0.},
    std::array{0., 0., iface.mass, 0., 0., 0.},
    std::array{
      0., 0., 0., iface.moments_of_inertia[0], iface.moments_of_inertia[1],
      iface.moments_of_inertia[2]},
    std::array{
      0., 0., 0., iface.moments_of_inertia[3], iface.moments_of_inertia[4],
      iface.moments_of_inertia[5]},
    std::array{
      0., 0., 0., iface.moments_of_inertia[6], iface.moments_of_inertia[7],
      iface.moments_of_inertia[8]}};

  const double damping_factor = iface.rho_inf;
  const int number_of_nonlinear_iterations =
    iface.number_of_nonlinear_iterations;

  kynema_fmb::interfaces::cfd::InterfaceInput point_to_build;
  point_to_build.gravity = gravity_;
  point_to_build.time_step = dtKynemaUGF;
  point_to_build.max_iter = number_of_nonlinear_iterations;
  point_to_build.rho_inf = damping_factor;
  point_to_build.turbine.floating_platform.enable = true;
  point_to_build.turbine.floating_platform.position = std::array<double, 7>{
    iface.center_of_mass[0] + iface.disp_init[0],
    iface.center_of_mass[1] + iface.disp_init[1],
    iface.center_of_mass[2] + iface.disp_init[2],
    iface.q_init[0],
    iface.q_init[1],
    iface.q_init[2],
    iface.q_init[3]};
  point_to_build.turbine.floating_platform.velocity = std::array<double, 6>{
    iface.v_init[0],     iface.v_init[1],     iface.v_init[2],
    iface.omega_init[0], iface.omega_init[1], iface.omega_init[2]};
  point_to_build.turbine.floating_platform.acceleration = std::array<double, 6>{
    iface.a_init[0],     iface.a_init[1],     iface.a_init[2],
    iface.alpha_init[0], iface.alpha_init[1], iface.alpha_init[2]};
  point_to_build.turbine.floating_platform.mass_matrix = mass_matrix;

  point_to_build.turbine.floating_platform.mooring_lines.resize(
    iface.tethers.size());

  for (int iteth = 0; iteth < iface.tethers.size(); ++iteth) {
    auto& tether = iface.tethers[iteth];
    auto&& mooring_line =
      point_to_build.turbine.floating_platform.mooring_lines[iteth];
    mooring_line.stiffness = tether.stiffness;
    mooring_line.undeformed_length = tether.initial_length;
    mooring_line.fairlead_position = tether.fairlead_position;
    mooring_line.anchor_position = tether.anchor_position;
  }

  point.bulk = bulk;
  iface.kynema_interface =
    std::make_shared<kynema_fmb::interfaces::cfd::Interface>(point_to_build);

  auto& meta = bulk->mesh_meta_data();

  point.total_force = meta.get_field<double>(meta.side_rank(), "tforce_scs");
  if (point.total_force == NULL)
    point.total_force =
      &(meta.declare_field<double>(meta.side_rank(), "tforce_scs"));

  for (const auto& surface_name : point.forcing_surface_names) {

    stk::mesh::Part* part = meta.get_part(surface_name);
    point.forcing_surfaces.push_back(part);

    const auto the_topo = part->topology();

    stk::mesh::put_field_on_mesh(
      *point.total_force, *part, 4 * 2 * meta.spatial_dimension(), nullptr);
  }

  for (const auto& block_name : point.moving_mesh_block_names) {
    stk::mesh::Part* part = meta.get_part(block_name);
    point.moving_mesh_blocks.push_back(part);
  }

  point.calc_loads = std::make_shared<CalcLoads>(point.forcing_surfaces);
  point.calc_loads->setup(bulk);
}
void
KynemaFMBSixDof::setup(
  double dtKynemaUGF, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  dt_ = dtKynemaUGF;
  for (int i = 0; i < (int)point_bodies_.size(); ++i) {
    setup_point(point_bodies_[i], point_interfaces_[i], dtKynemaUGF, bulk);
  }
}

void
KynemaFMBSixDof::initialize(int restartFreqKynemaUGF, double curTime)
{

  restart_frequency_ = restartFreqKynemaUGF;

  // Check for restart files and initialize values appropriately
  for (int ipoint = 0; ipoint < point_bodies_.size(); ipoint++) {
    if (point_interfaces_[ipoint].use_restart_data) {
      std::string file_name = std::to_string(ipoint) + "_" +
                              point_interfaces_[ipoint].restart_file_name;
      if (std::filesystem::exists(file_name)) {
        point_interfaces_[ipoint].kynema_interface->ReadRestart(file_name);
      }
    }
  }

  map_displacements(curTime, false);

  // Might not need to do this, need to evaluate
  if (curTime < 1e-10) {

    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << "Setting displacements at time steps n and n-1" << std::endl;

    auto& meta = bulk_->mesh_meta_data();

    const VectorFieldType* meshDisp =
      meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");
    const VectorFieldType* meshVel =
      meta.get_field<double>(stk::topology::NODE_RANK, "mesh_velocity");

    const VectorFieldType* meshDispNp1 =
      &(meshDisp->field_of_state(stk::mesh::StateNP1));
    VectorFieldType* meshDispN = &(meshDisp->field_of_state(stk::mesh::StateN));
    VectorFieldType* meshDispNm1 =
      &(meshDisp->field_of_state(stk::mesh::StateNM1));
    const VectorFieldType* meshVelNp1 =
      &(meshVel->field_of_state(stk::mesh::StateNP1));

    meshDisp->sync_to_host();
    meshVel->sync_to_host();
    meshDispNp1->sync_to_host();
    meshDispN->sync_to_host();
    meshDispNm1->sync_to_host();
    meshVelNp1->sync_to_host();

    stk::mesh::Selector sel = meta.universal_part();
    const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
    for (const auto* b : bkts) {
      for (const auto node : *b) {
        const double* velNp1 = stk::mesh::field_data(*meshVelNp1, node);
        const double* dispNp1 = stk::mesh::field_data(*meshDispNp1, node);
        double* dispN = stk::mesh::field_data(*meshDispN, node);
        double* dispNm1 = stk::mesh::field_data(*meshDispNm1, node);
        for (size_t i = 0; i < 3; i++) {
          dispN[i] = dispNp1[i] - dt_ * velNp1[i];
          dispNm1[i] = dispNp1[i] - 2.0 * dt_ * velNp1[i];
        }
      }
    }
    meshDispN->modify_on_host();
    meshDispNm1->modify_on_host();
  }
}

void
KynemaFMBSixDof::advance_struct_timestep(
  const double currentTime, const double dT)
{
  for (int ipoint = 0; ipoint < point_bodies_.size(); ++ipoint) {
    auto&& point = point_bodies_[ipoint];
    auto&& iface = point_interfaces_[ipoint];
    iface.kynema_interface->parameters.h = dT;
    auto converged = iface.kynema_interface->Step();

    if (!converged) {
      KynemaUGFEnv::self().kynema_ugfOutputP0()
        << "Kynema did not converge! Consider raising "
           "number_of_nonlinear_iterations for point body "
        << ipoint << std::endl;
    }

    if (
      (iface.kynema_interface->current_timestep_ % restart_frequency_) == 0 &&
      KynemaUGFEnv::self().parallel_rank() == 0) {
      std::string file_name =
        std::to_string(ipoint) + "_" + iface.restart_file_name;
      iface.kynema_interface->WriteRestart(file_name);
    }
    // Add output here
    if (
      iface.output_file_name.size() > 0 &&
      KynemaUGFEnv::self().parallel_rank() == 0) {
      std::string delim = " ";
      std::ofstream outfile(iface.output_file_name, std::ios::app);
      outfile << currentTime << delim;
      for (int idir = 0; idir < 7; ++idir)
        outfile << iface.kynema_interface->turbine.floating_platform.node
                     .position[idir]
                << delim;
      for (int idir = 0; idir < 6; ++idir)
        outfile << iface.kynema_interface->turbine.floating_platform.node
                     .velocity[idir]
                << delim;
      for (int idir = 0; idir < 5; ++idir)
        outfile
          << iface.kynema_interface->turbine.floating_platform.node.loads[idir]
          << delim;
      outfile << iface.kynema_interface->turbine.floating_platform.node.loads[5]
              << std::endl;
    }
  }
}

void
KynemaFMBSixDof::map_displacements(double current_time, bool updateCurCoor)
{
  for (int i = 0; i < (int)point_bodies_.size(); ++i) {
    point_bodies_[i].p_data.pos =
      point_interfaces_[i]
        .kynema_interface->turbine.floating_platform.node.position;
    point_bodies_[i].p_data.vel =
      point_interfaces_[i]
        .kynema_interface->turbine.floating_platform.node.velocity;
    map_displacements_point(point_bodies_[i], updateCurCoor);
  }
}

void
KynemaFMBSixDof::map_loads(const double)
{
  for (int i = 0; i < (int)point_bodies_.size(); ++i) {
    map_loads_point(point_bodies_[i]);

    auto& forces_and_moments = point_bodies_[i].p_data.loads;
    auto& iface = point_interfaces_[i];
    auto& loads = iface.kynema_interface->turbine.floating_platform.node.loads;
    for (int idim = 0; idim < 6; ++idim) {
      loads[idim] = 0.5 * (1.0 - iface.rho_inf) * forces_and_moments[idim] +
                    0.5 * (1.0 - iface.rho_inf) * loads[idim] +
                    loads[idim] * iface.rho_inf;
    }
  }
}

} // namespace kynema_ugf

} // namespace sierra
