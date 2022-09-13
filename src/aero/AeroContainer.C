// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/AeroContainer.h>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <yaml-cpp/yaml.h>

namespace sierra {
namespace nalu {

AeroContainer::AeroContainer(const YAML::Node& node)
{
  // look for Actuator
  std::vector<const YAML::Node*> foundActuator;
  NaluParsingHelper::find_nodes_given_key("actuator", node, foundActuator);
  if (foundActuator.size() > 0) {
    if (foundActuator.size() != 1)
      throw std::runtime_error(
        "look_ahead_and_create::error: Too many actuator line blocks");
    actuatorModel_.parse(*foundActuator[0]);
  }
}

void
AeroContainer::register_nodal_fields_fields(
  stk::mesh::MetaData& meta, stk::mesh::Part* part)
{
  if (has_actuators()) {
    VectorFieldType* actuatorSource = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source"));
    VectorFieldType* actuatorSourceLHS = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source_lhs"));
    stk::mesh::put_field_on_mesh(*actuatorSource, *part, nDim, nullptr);
    stk::mesh::put_field_on_mesh(*actuatorSourceLHS, *part, nDim, nullptr);
  }
}

void
AeroContainer::setup(double timeStep, stk::mesh::BulkData& bulk)
{
  if (has_actuators()) {
    actuatorModel_.setup(timeStep, bulk);
  }
}

void
AeroContainer::init(stk::mesh::BulkData& bulk)
{
  if (has_actuators) {
    actutatorModel_.init(bulk);
  }
}

void
AeroContainer::execute(double& actTimer)
{
  if (has_actuators()) {
    actuatorModel_.execute(actTimer);
  }
}

} // namespace nalu
} // namespace sierra
