// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "aero/six_dof/OpenTurbineSixDof.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include <fstream>
#include <NaluParsing.h>

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

namespace sierra {

namespace nalu {

OpenTurbineSixDof::OpenTurbineSixDof(const YAML::Node& node)
  : enable_calc_loads_(true)
{
  load(node);
}

void
OpenTurbineSixDof::load_point(const YAML::Node& node)
{

  PointMass new_body;
  const int ndim = 3;
  const int tensor_ndim = ndim * ndim;

  assert(node["moments_of_inertia"]);
  assert(node["moments_of_inertia"].size() == tensor_ndim);
  assert(node["center_of_mass"]);
  assert(node["center_of_mass"].size() == ndim);
  assert(node["mass"]);

  if (node["output_file_name"])
    new_body.output_file_name = node["output_file_name"].as<std::string>();

  if (node["use_restart_data"])
    new_body.use_restart_data = node["use_restart_data"].as<bool>();

  for (int d = 0; d < tensor_ndim; ++d) {
    new_body.moments_of_inertia[d] = node["moments_of_inertia"][d].as<double>();    
  }
  for (int d = 0; d < ndim; ++d) {
    new_body.center_of_mass[d] = node["center_of_mass"][d].as<double>();
  }

  new_body.mass = node["mass"].as<double>();

  if (node["forcing_surfaces"]) {
    for (std::size_t isurf = 0; isurf < node["forcing_surfaces"].size(); ++isurf) {
      new_body.forcing_surface_names.emplace_back(node["forcing_surfaces"][isurf].as<std::string>());
    }
  }

  if (node["moving_mesh_blocks"]) {
    for (std::size_t iblock = 0; iblock < node["moving_mesh_blocks"].size(); ++iblock) {
      new_body.moving_mesh_block_names.emplace_back(node["moving_mesh_blocks"][iblock].as<std::string>());
    }
  }

  if (node["tethers_initial_length"]    || 
      node["tethers_stiffness"]         || 
      node["tethers_fairlead_position"] ||
      node["tethers_anchor_position"]) {

    assert(node["tethers_initial_length"]);
    assert(node["tethers_stiffness"]);
    assert(node["tethers_fairlead_position"]);
    assert(node["tethers_anchor_position"]);

    assert(node["tethers_initial_length"].size() == node["tethers_stiffness"].size());
    assert(ndim * node["tethers_initial_length"].size() == node["tethers_fairlead_position"].size());
    assert(ndim * node["tethers_initial_length"].size() == node["tethers_anchor_position"].size());

    const int number_of_tethers = node["tethers_initial_length"].size();

    for (int itether = 0; itether < number_of_tethers; ++itether) {

      new_body.tethers.emplace_back(Tether());
      auto &&tether = new_body.tethers.back();

      tether.stiffness = node["tethers_stiffness"][itether].as<double>();
      tether.initial_length = node["tethers_initial_length"][itether].as<double>();
      for (int d = 0; d < ndim; ++d) {
        tether.fairlead_position[d] = node["tethers_fairlead_position"][3 * itether + d].as<double>();
        tether.anchor_position[d] = node["tethers_anchor_position"][3 * itether + d].as<double>();
      }
    }
    
  }
  point_bodies_.emplace_back(new_body);

}

void
OpenTurbineSixDof::load(const YAML::Node& node)
{
  const int ndim = 3;
  get_required(node, "number_of_bodies", number_of_bodies_);

  if(node["gravity"]) {
    for (int idim = 0; idim < ndim; ++idim) {
      gravity_[idim] = node["gravity"][idim].as<double>();
    }
  }

  for (int ibody = 0; ibody < number_of_bodies_; ++ibody) {
    if (!node["Body" + std::to_string(ibody)]) {
      throw std::runtime_error("Node for Body" + std::to_string(ibody) + "not present or correct in input file");
    }

    auto body_node = node["Body" + std::to_string(ibody)];

    assert(body_node["type"]);

    std::string body_type = body_node["type"].as<std::string>();

    if (body_type == "point") {
      load_point(body_node);
    } else {
      throw std::runtime_error("unrecognized body type for 6 DOF. Currently only point is supported.");
    }
  } 

}

void
OpenTurbineSixDof::setup_point(PointMass &point, const double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk)
{

  auto mass_matrix = std::array{
    std::array{point.mass, 0., 0., 0., 0., 0.},
    std::array{0., point.mass, 0., 0., 0., 0.},
    std::array{0., 0., point.mass, 0., 0., 0.},
    std::array{0., 0., 0., point.moments_of_inertia[0], point.moments_of_inertia[1], point.moments_of_inertia[2]},
    std::array{0., 0., 0., point.moments_of_inertia[3], point.moments_of_inertia[4], point.moments_of_inertia[5]},
    std::array{0., 0., 0., point.moments_of_inertia[6], point.moments_of_inertia[7], point.moments_of_inertia[8]}
  };

  // Sticking to 5 nonlinear iterations and no-damping to match working example and avoid user knobs.
  constexpr double damping_factor = 0.0;
  constexpr int number_of_nonlinear_iterations = 5;

  openturbine::cfd::InterfaceInput point_to_build;
  point_to_build.gravity = gravity_;
  point_to_build.time_step = dtNalu;
  point_to_build.max_iter = number_of_nonlinear_iterations;
  point_to_build.rho_inf = damping_factor;
  point_to_build.turbine.floating_platform.enable = true;
  point_to_build.turbine.floating_platform.position = std::array<double,7>{point.center_of_mass[0], point.center_of_mass[1], point.center_of_mass[2], 1.0, 0.0, 0.0, 0.0};
  point_to_build.turbine.floating_platform.mass_matrix = mass_matrix;

  point_to_build.turbine.floating_platform.mooring_lines.resize(point.tethers.size());

  for (int iteth = 0; iteth < point.tethers.size(); ++iteth) {
    auto& tether = point.tethers[iteth];
    auto&& mooring_line = point_to_build.turbine.floating_platform.mooring_lines[iteth];
    mooring_line.stiffness = tether.stiffness;
    mooring_line.undeformed_length = tether.initial_length;
    mooring_line.fairlead_position = tether.fairlead_position;
    mooring_line.anchor_position   = tether.anchor_position;
  }

  point.bulk = bulk;
  point.openturbine_interface = std::make_shared<openturbine::cfd::Interface>(point_to_build);

  auto& meta = bulk->mesh_meta_data();

  point.total_force = meta.get_field<double>(meta.side_rank(), "tforce_scs");
  if (point.total_force == NULL)
    point.total_force = &(meta.declare_field<double>(meta.side_rank(), "tforce_scs"));

  for (const auto & surface_name : point.forcing_surface_names) {

    stk::mesh::Part* part = meta.get_part(surface_name);
    point.forcing_surfaces.push_back(part);
    
    const auto the_topo = part->topology();
    //auto me_fc = MasterElementRepo::get_surface_master_element_on_host(the_topo);
    //const int numScsIp = me_fc->num_integration_points();

    stk::mesh::put_field_on_mesh(*point.total_force, *part, 4 * 2 * meta.spatial_dimension(), nullptr);
  }

  for (const auto & block_name : point.moving_mesh_block_names) {
    stk::mesh::Part* part = meta.get_part(block_name);
    point.moving_mesh_blocks.push_back(part);
  }

  point.calc_loads = std::make_shared<CalcLoads>(point.forcing_surfaces);
  point.calc_loads->setup(bulk);

}
void
OpenTurbineSixDof::setup(double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  dt_ = dtNalu;
  for (auto& point : point_bodies_) {
    setup_point(point, dtNalu, bulk);
  }
}

void
OpenTurbineSixDof::initialize(int restartFreqNalu, double curTime)
{

  restart_frequency_ = restartFreqNalu;

  // Check for restart files and initialize values appropriately 
  for (int ipoint = 0; ipoint < point_bodies_.size(); ipoint++) {
    if (point_bodies_[ipoint].use_restart_data) {
      std::string file_name = std::to_string(ipoint) + "_" + point_bodies_[ipoint].restart_file_name;
      if (std::filesystem::exists(file_name)) {
        point_bodies_[ipoint].openturbine_interface->ReadRestart(file_name);
      }
    }
  }

  map_displacements(curTime, false);

  // Might not need to do this, need to evaluate
  if (curTime < 1e-10) {

    NaluEnv::self().naluOutputP0()
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
OpenTurbineSixDof::advance_struct_timestep(const double currentTime, const double dT)
{
  for (int ipoint = 0; ipoint < point_bodies_.size(); ++ipoint) {
    auto && point = point_bodies_[ipoint];
    point.openturbine_interface->parameters.h = dT;
    auto _converged = point.openturbine_interface->Step();
    if ((point.openturbine_interface->current_timestep_ % restart_frequency_) == 0) {
      std::string file_name = std::to_string(ipoint) + "_" + point.restart_file_name;
      point.openturbine_interface->WriteRestart(file_name);
    }
    // Add output here
    if (point.output_file_name.size() > 0 && NaluEnv::self().parallel_rank() == 0) {
      std::string delim = " ";
      std::ofstream outfile(point.output_file_name, std::ios::app);
      outfile << currentTime << delim;
      for (int idir = 0; idir < 7; ++idir) 
        outfile << point.openturbine_interface->turbine.floating_platform.node.position[idir] << delim;
      for (int idir = 0; idir < 6; ++idir) 
        outfile << point.openturbine_interface->turbine.floating_platform.node.velocity[idir] << delim;
      for (int idir = 0; idir < 5; ++idir) 
        outfile << point.openturbine_interface->turbine.floating_platform.node.loads[idir] << delim;
      outfile << point.openturbine_interface->turbine.floating_platform.node.loads[5] << std::endl;
    } 
  }
}

void
OpenTurbineSixDof::map_displacements_point(PointMass &point, bool updateCur)
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
 
  std::array<double, 7> translation_and_rotation_position = point.openturbine_interface->turbine.floating_platform.node.position;
  std::array<double, 6> translation_and_rotation_velocities = point.openturbine_interface->turbine.floating_platform.node.velocity;
  std::array<double, 3> current_center_of_mass_location = {
    translation_and_rotation_position[0],
    translation_and_rotation_position[1],
    translation_and_rotation_position[2]};

  auto q0 = translation_and_rotation_position[3];
  auto q1 = translation_and_rotation_position[4];
  auto q2 = translation_and_rotation_position[5];
  auto q3 = translation_and_rotation_position[6];

  std::array<std::array<double, 3>, 3> current_rotation_matrix = 
    {{{q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q0 * q2 + q1 * q3)},
     {2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 - q0 * q1)},
     {2.0 * (q1 * q3 - q0 * q2),2.0 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3}}};

  std::array<double, 3> new_point = {0.0, 0.0, 0.0};
  std::array<double, 3> current_point = {0.0, 0.0, 0.0};
  std::array<double, 3> new_velocity = {0.0, 0.0, 0.0};
  std::array<double, 3> lever_arm = {0.0, 0.0, 0.0};

  stk::mesh::Selector sel(stk::mesh::selectUnion(point.moving_mesh_blocks));
  const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);

  auto cross_product = [](double* a, double* b, double* axb)
  {
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
        current_point[row] = modelc[row] - point.center_of_mass[row];
      }

      for (int row = 0; row < 3; ++row) {
        new_point[row] = 0.0;
        for (int col = 0; col < 3; ++col) {
          new_point[row] += current_rotation_matrix[row][col] * current_point[col];
        }
        new_point[row] += current_center_of_mass_location[row];
      }


      for (int row = 0; row < 3; ++row) {
        disp[row] = new_point[row] - modelc[row];
        lever_arm[row] = new_point[row] - current_center_of_mass_location[row];
      }

      cross_product(&translation_and_rotation_velocities[3], lever_arm.data(), new_velocity.data());

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
OpenTurbineSixDof::map_displacements(double current_time, bool updateCurCoor)
{
  for (auto& point : point_bodies_) {
    map_displacements_point(point, updateCurCoor);
  }
}

void
OpenTurbineSixDof::map_loads_point(PointMass &point)
{
  point.calc_loads->initialize();
  point.calc_loads->execute();

  auto& meta = bulk_->mesh_meta_data();
  const VectorFieldType* modelCoords =
    meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
  const VectorFieldType* meshDisp =
    meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");

  std::array<double, 7> translation_and_rotation_position = 
   point.openturbine_interface->turbine.floating_platform.node.position;

  std::array<double, 3> center_of_mass = { 
    translation_and_rotation_position[0], 
    translation_and_rotation_position[1], 
    translation_and_rotation_position[2]};

  auto forces_and_moments = 
    fsi::accumulateLoadsAndMoments(*bulk_, point.forcing_surfaces, 
      *modelCoords, *meshDisp, *(point.total_force), center_of_mass);

  // Reduce to get full result and then feed into open turbine
  MPI_Allreduce(MPI_IN_PLACE, forces_and_moments.data(), 6, MPI_DOUBLE, MPI_SUM, bulk_->parallel());

  for (int idim = 0; idim < 6; ++idim) {
    point.openturbine_interface->turbine.floating_platform.node.loads[idim] = 0.5 * forces_and_moments[idim] +
      0.5 * point.openturbine_interface->turbine.floating_platform.node.loads[idim];
  }

}

void
OpenTurbineSixDof::map_loads()
{
  for (auto &point : point_bodies_) {
    map_loads_point(point);
  }
}

} // namespace nalu

} // namespace sierra
