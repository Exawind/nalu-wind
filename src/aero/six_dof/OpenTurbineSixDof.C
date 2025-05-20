// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "aero/six_dof/OpenTurbineSixDof.h"
#include <NaluParsing.h>

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

  for (int d = 0; d < tensor_ndim; ++d) {
    new_body.moments_of_inertia[d] = node["moments_of_inertia"][d].as<double>();    
  }
  for (int d = 0; d < ndim; ++d) {
    new_body.center_of_mass[d] = node["center_of_mass"][d].as<double>();
  }

  new_body.mass = node["mass"].as<double>();

  if (node["forcing_surfaces"]) {
    for (std::size_t isurf = 0; isurf < node["forcing_surfaces"].size(); ++isurf) {
      new_body.forcing_surfaces.emplace_back(node["forcing_surfaces"][isurf].as<std::string>());
    }
  }

  if (node["moving_mesh_blocks"]) {
    for (std::size_t iblock = 0; iblock < node["moving_mesh_blocks"].size(); ++iblock) {
      new_body.moving_mesh_blocks.emplace_back(node["moving_mesh_blocks"][iblock].as<std::string>());
    }
  }

  if (node["tethers_initial_length"]    || 
      node["tethers_stiffness"]         || 
      node["tethers_fairlead_position"] ||
      node["tehers_anchor_position"]) {

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

}

void
OpenTurbineSixDof::load(const YAML::Node& node)
{

  const int ndim = 3;
  get_required(node, "number_of_bodies", number_of_bodies_);

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
OpenTurbineSixDof::setup(double dtNalu, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  dt_ = dtNalu;

  // Build OpenTurbine "platform here"
}

void
OpenTurbineSixDof::initialize(int restartFreqNalu, double curTime)
{

  compute_mapping();

  // Init open turbine values here accounting for restart

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
OpenTurbineSixDof::compute_mapping()
{
}

void
OpenTurbineSixDof::predict_struct_states()
{
}

void
OpenTurbineSixDof::predict_struct_timestep(const double curTime)
{
}

void
OpenTurbineSixDof::advance_struct_timestep(const double /* curTime */)
{
  // advance OpenTurbine solve for each body
  // interface.Step();
}

void
OpenTurbineSixDof::send_loads(const double /* curTime */)
{
  // Send forces to OpenTurbine for each body
  // interface.turbine.floating_platform.node.loads[0->6] = forces then moments 
}

void
OpenTurbineSixDof::get_displacements(double /* current_time */)
{
  // Pull translation and rotation from OpenTurbine store to each body
  // interface.turbine.floating_platform.node.displacement (3 points for COM, 4 for quaternion) 
}

void
OpenTurbineSixDof::compute_div_mesh_velocity()
{
  // Should always be 0.0 if motion done correctly
}

void
OpenTurbineSixDof::map_displacements(double current_time, bool updateCurCoor)
{

  get_displacements(current_time);

  // Should do something very similar to what is done below.
  // Note we do not need a "map" really as we operate on volumes currently

  //stk::mesh::Selector sel;
  //int nTurbinesGlob = FAST.get_nTurbinesGlob();
  //for (int i = 0; i < nTurbinesGlob; i++) {
  //  if (fsiTurbineData_[i] != NULL) {
  //    fsiTurbineData_[i]->mapDisplacements(current_time);
  //    sel &= stk::mesh::selectUnion(fsiTurbineData_[i]->getPartVec());
  //  }
  //}

  //if (updateCurCoor) {
  //  auto& meta = bulk_->mesh_meta_data();
  //  const VectorFieldType* modelCoords =
  //    meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
  //  VectorFieldType* curCoords =
  //    meta.get_field<double>(stk::topology::NODE_RANK, "current_coordinates");
  //  VectorFieldType* displacement =
  //    meta.get_field<double>(stk::topology::NODE_RANK, "mesh_displacement");

  //  modelCoords->sync_to_host();
  //  curCoords->sync_to_host();
  //  displacement->sync_to_host();

  //  const auto& bkts = bulk_->get_buckets(stk::topology::NODE_RANK, sel);
  //  for (const auto* b : bkts) {
  //    for (const auto node : *b) {
  //      for (size_t in = 0; in < b->size(); in++) {

  //        double* cc = stk::mesh::field_data(*curCoords, node);
  //        double* mc = stk::mesh::field_data(*modelCoords, node);
  //        double* cd = stk::mesh::field_data(*displacement, node);

  //        for (int j = 0; j < 3; ++j) {
  //          cc[j] = mc[j] + cd[j];
  //        }
  //      }
  //    }
  //  }

  //  curCoords->modify_on_host();
  //  curCoords->sync_to_device();
  //}
  //timer_stop(naluTimer_);
}

void
OpenTurbineSixDof::map_loads(const int tStep, const double curTime)
{
  //
  // Reduce loads and store
  
  //timer_start(naluTimer_);
  //int nTurbinesGlob = FAST.get_nTurbinesGlob();
  //for (int i = 0; i < nTurbinesGlob; i++) {
  //  if (fsiTurbineData_[i] != nullptr) { // This may not be a turbine intended
  //                                       // for blade-resolved simulation
  //    int turbProc = fsiTurbineData_[i]->getProc();
  //    fsiTurbineData_[i]->mapLoads();
  //    int nTotBldNodes = fsiTurbineData_[i]->params_.nTotBRfsiPtsBlade;
  //    if (bulk_->parallel_rank() == turbProc) {
  //      MPI_Reduce(
  //        MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.twr_ld.data(),
  //        (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, MPI_SUM,
  //        turbProc, bulk_->parallel());
  //      MPI_Reduce(
  //        MPI_IN_PLACE, fsiTurbineData_[i]->brFSIdata_.bld_ld.data(),
  //        nTotBldNodes * 6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_->parallel());
  //    } else {
  //      MPI_Reduce(
  //        fsiTurbineData_[i]->brFSIdata_.twr_ld.data(), NULL,
  //        (fsiTurbineData_[i]->params_.nBRfsiPtsTwr) * 6, MPI_DOUBLE, MPI_SUM,
  //        turbProc, bulk_->parallel());
  //      MPI_Reduce(
  //        fsiTurbineData_[i]->brFSIdata_.bld_ld.data(), NULL,
  //        (nTotBldNodes) * 6, MPI_DOUBLE, MPI_SUM, turbProc, bulk_->parallel());
  //    }

  //    fsiTurbineData_[i]->write_nc_def_loads(tStep, curTime);
  //  }
  //}
  //timer_stop(naluTimer_);
}

} // namespace nalu

} // namespace sierra
