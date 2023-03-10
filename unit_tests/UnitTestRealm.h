// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTREALM_H
#define UNITTESTREALM_H

#include "Simulation.h"
#include "Realm.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include "yaml-cpp/yaml.h"

namespace unit_test_utils {

YAML::Node get_default_inputs();

YAML::Node get_realm_default_node();

class NaluTest
{
public:
  NaluTest(const YAML::Node& doc = get_default_inputs());

  ~NaluTest();

  sierra::nalu::Realm& create_realm(
    const YAML::Node& realm_node = get_realm_default_node(),
    const std::string realm_type = "multi_physics",
    const bool createMeshObjects = true);

  YAML::Node doc_;
  stk::ParallelMachine comm_;
  unsigned spatialDim_;
  sierra::nalu::Simulation sim_;

  stk::mesh::PartVector partVec_;

private:
  NaluTest(const NaluTest&) = delete;
  std::string logFileName_;
};

} // namespace unit_test_utils

#endif /* UNITTESTREALM_H */
