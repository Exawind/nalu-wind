// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
// This file is only compiled when ENABLE_KYNEMA_FMB_SIXDOF is on.
#include <aero/KynemaContainer.h>
#include <KynemaUGFEnv.h>
#include "aero/fmb/KynemaFMBSixDof.h"

namespace sierra {
namespace kynema_ugf {

KynemaContainer::~KynemaContainer() = default;

KynemaContainer::KynemaContainer(const YAML::Node& node)
{
  if (node["kynema_fmb_six_dof"]) {
    sixDof_ = std::make_shared<KynemaFMBSixDof>(node["kynema_fmb_six_dof"]);
  }
}

void
KynemaContainer::setup(
  double timeStep, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  if (has_six_dof()) {
    sixDof_->setup(timeStep, bulk_);
  }
}

void
KynemaContainer::init(double currentTime, double restartFrequency)
{
  if (has_six_dof()) {
    sixDof_->initialize(restartFrequency, currentTime);
  }
}

void
KynemaContainer::update_displacements(const double currentTime, bool updateCC)
{
  if (has_six_dof()) {
    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << "Calling update displacements inside KynemaContainer" << std::endl;
    sixDof_->map_displacements(currentTime, updateCC);
  }
}

void
KynemaContainer::predict_model_time_step(const double currentTime)
{
  if (has_six_dof()) {
    sixDof_->map_loads(currentTime);
  }
}

void
KynemaContainer::advance_model_time_step(
  const double currentTime, const double dT)
{
  if (has_six_dof()) {
    sixDof_->advance_struct_timestep(currentTime, dT);
  }
}

const stk::mesh::PartVector
KynemaContainer::six_dof_parts()
{
  if (has_six_dof()) {
    return sixDof_->get_mesh_blocks();
  }
  return stk::mesh::PartVector{};
}

} // namespace kynema_ugf
} // namespace sierra
