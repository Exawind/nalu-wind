#ifndef FMBCONTAINER_H_
#define FMBCONTAINER_H_

#include <memory>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <yaml-cpp/yaml.h>

namespace sierra::kynema_ugf {
class KynemaFMBSixDof;

/**
 * A container class for holding all the Kynema-FMB structural models
 * (six-dof bodies, etc)
 */
class FMBContainer
{
public:
  FMBContainer() = delete;
  FMBContainer operator=(FMBContainer&) = delete;
  FMBContainer(FMBContainer&) = delete;

  FMBContainer(const YAML::Node& node);
  ~FMBContainer();

  void setup(double timeStep, std::shared_ptr<stk::mesh::BulkData> stkBulk);
  void init(double currentTime, double restartFrequency);
  void update_displacements(
    const double currentTime, const bool updateCurCoords = true);
  void predict_model_time_step(const double /*currentTime*/);
  void advance_model_time_step(const double /*currentTime*/, const double);

  bool is_active() { return has_six_dof(); }
  bool has_six_dof() { return sixDof_ != nullptr; }

  const stk::mesh::PartVector six_dof_parts();

private:
  std::shared_ptr<KynemaFMBSixDof> sixDof_;
  std::shared_ptr<stk::mesh::BulkData> bulk_;
};

} // namespace sierra::kynema_ugf
#endif
