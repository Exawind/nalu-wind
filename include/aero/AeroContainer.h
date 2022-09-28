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

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <yaml-cpp/yaml.h>
#include "aero/actuator/ActuatorModel.h"

#ifdef NALU_USES_OPENFAST
#include "OpenFAST.H"
// TODO add a blank openfast data structure possibly
#endif

namespace sierra {
namespace nalu {

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
  ~AeroContainer() = default;

  void setup(double timeStep, stk::mesh::BulkData& stkBulk);
  void execute(double& timer);
  void init(stk::mesh::BulkData& stkBulk);
  void register_nodal_fields(stk::mesh::MetaData& meta, stk::mesh::Part* part);

  // TODO active if actuators or FSI is active
  bool is_active() { return has_actuators(); }

private:
  bool has_actuators() { return actuatorModel_.is_active(); }
#ifdef NALU_USES_OPENFAST
  // TODO this should be the only instance of openfast
  // all other instances need to be made into shared_ptrs that share from this
  // one
  std::shared_ptr<fast::OpenFAST> fast_;
#endif
  // TODO move calls to actuator model in realm to this container class
  ActuatorModel actuatorModel_;
};

} // namespace nalu
} // namespace sierra
#endif
