// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORMODEL_H_
#define ACTUATORMODEL_H_

#include <memory>
#include <stdexcept>
#include <aero/actuator/ActuatorExecutor.h>
#include <aero/actuator/ActuatorFLLC.h>
#include <aero/actuator/ActuatorBulk.h>

namespace YAML {
class Node;
}

namespace stk {
namespace mesh {
class BulkData;
}
} // namespace stk

namespace dcast {
template <typename IN, typename OUT>
OUT*
dcast_and_check_pointer(IN* input)
{
  auto out = dynamic_cast<OUT*>(input);
  if (out == nullptr)
    throw std::runtime_error("dynamic cast failed");
  return out;
}
} // namespace dcast

namespace sierra {
namespace nalu {

/**
 * @brief This is the interface class for running all the kokkos based actuator
 * models
 *
 * This interface class should handle the logic for construction and execution
 * of the underlying models at the highest abstraction levels possible since we
 * only allow one actuator section per input file we can consider it a singelton
 */
struct ActuatorModel
{
  std::shared_ptr<ActuatorMeta> actMeta_;
  std::shared_ptr<ActuatorBulk> actBulk_;
  std::shared_ptr<ActuatorExecutor> actExec_;

  ActuatorModel() = default;
  virtual ~ActuatorModel(){};

  void parse(const YAML::Node& actuatorNode);
  void setup(double timeStep, stk::mesh::BulkData& stkBulk);
  void execute(double& timer);
  void init(stk::mesh::BulkData& stkBulk);
  inline bool is_active() { return actMeta_ != nullptr; }
};

} // namespace nalu
} // namespace sierra

#endif /* ACTUATORMODEL_H_ */
