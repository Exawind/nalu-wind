// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include <aero/KynemaContainer.h>
#include <KynemaUGFEnv.h>
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
#include "aero/fmb/KynemaFMBSixDof.h"
#endif

namespace sierra {
namespace kynema_ugf {

KynemaContainer::~KynemaContainer() = default;

KynemaContainer::KynemaContainer(const YAML::Node& node)
{
  if (node["kynema_fmb_six_dof"]) {
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
    sixDof_ = std::make_shared<KynemaFMBSixDof>(node["kynema_fmb_six_dof"]);
#else
    throw std::runtime_error(
      "6DOF coupling can not be used without coupling to Kynema-FMB");
#endif
  }
}

void
KynemaContainer::setup(
  double timeStep, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
  if (has_six_dof()) {
    sixDof_->setup(timeStep, bulk_);
  }
#else
  (void)timeStep;
#endif
}

void
KynemaContainer::init(double currentTime, double restartFrequency)
{
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
  if (has_six_dof()) {
    sixDof_->initialize(restartFrequency, currentTime);
  }
#else
  (void)currentTime;
  (void)restartFrequency;
#endif
}

void
KynemaContainer::update_displacements(const double currentTime, bool updateCC)
{
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
  if (has_six_dof()) {
    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << "Calling update displacements inside KynemaContainer" << std::endl;
    sixDof_->map_displacements(currentTime, updateCC);
  }
#else
  (void)currentTime;
  (void)updateCC;
#endif
}

void
KynemaContainer::predict_model_time_step(const double currentTime)
{
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
  if (has_six_dof()) {
    sixDof_->map_loads(currentTime);
  }
#else
  (void)currentTime;
#endif
}

void
KynemaContainer::advance_model_time_step(
  const double
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
    currentTime
#endif
  ,
  const double
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
    dT
#endif
)
{
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
  if (has_six_dof()) {
    sixDof_->advance_struct_timestep(currentTime, dT);
  }
#endif
}

const stk::mesh::PartVector
KynemaContainer::six_dof_parts()
{
#ifdef KYNEMA_UGF_USES_KYNEMA_FMB
  if (has_six_dof()) {
    return sixDof_->get_mesh_blocks();
  }
#endif
  stk::mesh::PartVector all_part_vec;
  return all_part_vec;
}

} // namespace kynema_ugf
} // namespace sierra
