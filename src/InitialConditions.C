// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <InitialConditions.h>
#include <NaluEnv.h>
#include <Realm.h>

// yaml for parsing..
#include <yaml-cpp/yaml.h>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

std::unique_ptr<InitialCondition>
InitialConditionCreator::load_single(const YAML::Node& node)
{
  if (node["constant"]) {
    NaluEnv::self().naluOutputP0() << "Initial Is Type constant " << std::endl;
    std::unique_ptr<InitialCondition> ic =
      std::make_unique<ConstantInitialConditionData>(debug_);
    auto* constIC = dynamic_cast<ConstantInitialConditionData*>(ic.get());
    node >> *constIC;
    return ic;
  } else if (node["user_function"]) {
    NaluEnv::self().naluOutputP0()
      << "Initial Is Type user-function " << std::endl;
    std::unique_ptr<InitialCondition> ic =
      std::make_unique<UserFunctionInitialConditionData>();
    auto* fcnIC = dynamic_cast<UserFunctionInitialConditionData*>(ic.get());
    node >> *fcnIC;
    return ic;
  } else
    throw std::runtime_error(
      "parser error InitialConditions::load; unsupported IC type");
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
  return 0;
#endif
}

InitialConditionVector
InitialConditionCreator::create_ic_vector(const YAML::Node& node)
{
  InitialConditionVector vec;

  if (node["initial_conditions"]) {
    const YAML::Node initial_conditions = node["initial_conditions"];
    for (size_t j_initial_condition = 0;
         j_initial_condition < initial_conditions.size();
         ++j_initial_condition) {
      const YAML::Node initial_condition_node =
        initial_conditions[j_initial_condition];
      vec.push_back(load_single(initial_condition_node));
    }
  } else {
    throw std::runtime_error("parser error InitialConditions::load");
  }

  return vec;
}

} // namespace nalu
} // namespace sierra
