// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef InitialConditions_h
#define InitialConditions_h

#include <Enums.h>

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

class Realm;
class Simulation;

class InitialCondition
{
public:
  InitialCondition() : theIcType_(UserDataType_END) {}

  virtual ~InitialCondition() {}

  std::string icName_;
  std::vector<std::string> targetNames_;
  UserDataType theIcType_;
};

typedef std::vector<std::unique_ptr<InitialCondition>> InitialConditionVector;

struct InitialConditionCreator
{
  InitialConditionCreator(bool debug) : debug_(debug) {}
  InitialConditionVector create_ic_vector(const YAML::Node& node);
  std::unique_ptr<InitialCondition> load_single(const YAML::Node& node);
  const bool debug_;
};

} // namespace nalu
} // namespace sierra

#endif
