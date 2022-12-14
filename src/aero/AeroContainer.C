// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/AeroContainer.h>
#include <NaluParsingHelper.h>
#include "aero/fsi/OpenfastFSI.h"
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {
void
AeroContainer::clean_up()
{
  if (has_fsi())
    fsiContainer_->end_openfast();
}

AeroContainer::~AeroContainer()
{
  if (has_fsi()) {
    delete fsiContainer_;
  }
}

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
  std::vector<const YAML::Node*> foundFsi;
  NaluParsingHelper::find_nodes_given_key("openfast_fsi", node, foundFsi);
  if (foundFsi.size() > 0) {
    if (foundFsi.size() != 1)
      throw std::runtime_error(
        "look_ahead_and_create::error: Too many openfast_fsi blocks");
    fsiContainer_ = new OpenfastFSI(*foundFsi[0]);
  }
}

void
AeroContainer::register_nodal_fields(
  stk::mesh::MetaData& meta, stk::mesh::Part* part)
{
  if (has_actuators()) {
    const int nDim = meta.spatial_dimension();
    VectorFieldType* actuatorSource = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source"));
    VectorFieldType* actuatorSourceLHS = &(meta.declare_field<VectorFieldType>(
      stk::topology::NODE_RANK, "actuator_source_lhs"));
    stk::mesh::put_field_on_mesh(*actuatorSource, *part, nDim, nullptr);
    stk::mesh::put_field_on_mesh(*actuatorSourceLHS, *part, nDim, nullptr);
  }
}

void
AeroContainer::setup(double timeStep, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  if (has_actuators()) {
    actuatorModel_.setup(timeStep, *bulk_);
  }
  if (has_fsi()) {
    fsiContainer_->setup(timeStep, bulk_);
  }
}

void
AeroContainer::init(double currentTime, double restartFrequency)
{
  if (has_actuators()) {
    actuatorModel_.init(*bulk_);
  }
  if (has_fsi()) {
    fsiContainer_->initialize(restartFrequency, currentTime);
  }
}

void
AeroContainer::execute(double& actTimer)
{
  if (has_actuators()) {
    actuatorModel_.execute(actTimer);
  }
}
void
AeroContainer::update_displacements(const double currentTime)
{
  if (has_fsi()) {
    fsiContainer_->predict_struct_states();
    fsiContainer_->get_displacements(currentTime);
  }
}

void
AeroContainer::predict_model_time_step(const double currentTime)
{
  if (has_fsi()) {
    fsiContainer_->predict_struct_timestep(currentTime);
  }
}

void
AeroContainer::advance_model_time_step(const double currentTime)
{
  if (has_fsi()) {
    fsiContainer_->advance_struct_timestep(currentTime);
  }
}

void
AeroContainer::compute_div_mesh_velocity()
{
  if (has_fsi()) {
    fsiContainer_->compute_div_mesh_velocity();
  }
}
} // namespace nalu
} // namespace sierra
