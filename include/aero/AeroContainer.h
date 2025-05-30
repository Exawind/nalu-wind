// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AEROCONTAINER_H_
#define AEROCONTAINER_H_

#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <yaml-cpp/yaml.h>
#include "aero/actuator/ActuatorModel.h"
#include "aero/six_dof/OpenTurbineSixDof.h"

namespace sierra::nalu {
class OpenfastFSI;

/**
 * A container class for holding all the aerodynamic models (actuators,
 * fsi-turbines, etc)
 */
class AeroContainer
{
public:
  AeroContainer() = delete;
  AeroContainer operator=(AeroContainer&) = delete;
  AeroContainer(AeroContainer&) = delete;

  AeroContainer(const YAML::Node& node);
  ~AeroContainer();

  void setup(double timeStep, std::shared_ptr<stk::mesh::BulkData> stkBulk);
  void execute(double& timer);
  void init(double currentTime, double restartFrequency);
  void register_nodal_fields(
    stk::mesh::MetaData& meta, const stk::mesh::PartVector& part_vec);
  void update_displacements(
    const double currentTime,
    const bool updateCurCoords = true,
    const bool predict = true);
  void predict_model_time_step(const double /*currentTime*/);
  void advance_model_time_step(const double /*currentTime*/);
  void compute_div_mesh_velocity();
  // hacky function to make sure openfast is cleaned up
  // eventually all openfast pointers should be combined and moved out of this
  // class
  void clean_up();

  bool is_active() { return has_actuators() || has_fsi(); }
  bool has_fsi() { return fsiContainer_ != nullptr; }
  bool has_six_dof() {return sixDof_ != nullptr;}

  const stk::mesh::PartVector six_dof_parts();
  const stk::mesh::PartVector fsi_parts();
  const stk::mesh::PartVector fsi_bndry_parts();
  const std::vector<std::string> fsi_bndry_part_names();
  double openfast_accumulated_time();
  double nalu_fsi_accumulated_time();

private:
  bool has_actuators() { return actuatorModel_.is_active(); }

  bool has_six_dof_;
  ActuatorModel actuatorModel_;
  // TODO make this a unique_ptr
  OpenfastFSI* fsiContainer_;
  std::shared_ptr<OpenTurbineSixDof> sixDof_;
  std::shared_ptr<stk::mesh::BulkData> bulk_;
};

} // namespace sierra::nalu
#endif
