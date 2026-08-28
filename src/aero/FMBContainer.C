// This file is only compiled when ENABLE_KYNEMA_FMB_SIXDOF is on.
#include <aero/FMBContainer.h>
#include <KynemaUGFEnv.h>
#include "aero/fmb/KynemaFMBSixDof.h"

namespace sierra {
namespace kynema_ugf {

FMBContainer::~FMBContainer() = default;

FMBContainer::FMBContainer(const YAML::Node& node)
{
  if (node["kynema_fmb_six_dof"]) {
    sixDof_ = std::make_shared<KynemaFMBSixDof>(node["kynema_fmb_six_dof"]);
  }
}

void
FMBContainer::setup(double timeStep, std::shared_ptr<stk::mesh::BulkData> bulk)
{
  bulk_ = bulk;
  if (has_six_dof()) {
    sixDof_->setup(timeStep, bulk_);
  }
}

void
FMBContainer::init(double currentTime, double restartFrequency)
{
  if (has_six_dof()) {
    sixDof_->initialize(restartFrequency, currentTime);
  }
}

void
FMBContainer::update_displacements(const double currentTime, bool updateCC)
{
  if (has_six_dof()) {
    KynemaUGFEnv::self().kynema_ugfOutputP0()
      << "Calling update displacements inside FMBContainer" << std::endl;
    sixDof_->map_displacements(currentTime, updateCC);
  }
}

void
FMBContainer::predict_model_time_step(const double currentTime)
{
  if (has_six_dof()) {
    sixDof_->map_loads(currentTime);
  }
}

void
FMBContainer::advance_model_time_step(const double currentTime, const double dT)
{
  if (has_six_dof()) {
    sixDof_->advance_struct_timestep(currentTime, dT);
  }
}

const stk::mesh::PartVector
FMBContainer::six_dof_parts()
{
  if (has_six_dof()) {
    return sixDof_->get_mesh_blocks();
  }
  return stk::mesh::PartVector{};
}

} // namespace kynema_ugf
} // namespace sierra
