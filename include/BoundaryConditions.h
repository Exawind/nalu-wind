// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef BoundaryConditions_h
#define BoundaryConditions_h

#include <Enums.h>

#include <map>
#include <string>
#include <vector>
#include <memory>

namespace YAML {
  class Node;
}

namespace sierra{
  namespace nalu{

class Realm;
class BoundaryConditions;
class Simulation;

class BoundaryCondition {
 public:
 BoundaryCondition(BoundaryConditions& bcs) : boundaryConditions_(bcs) {}
  
  virtual ~BoundaryCondition() {}

  std::unique_ptr<BoundaryCondition> load(const YAML::Node& node);

  std::string bcName_;
  std::string targetName_;
  BoundaryConditionType theBcType_;
  BoundaryConditions& boundaryConditions_;
};

typedef std::vector<std::unique_ptr<BoundaryCondition>> BoundaryConditionVector;

class BoundaryConditions
{
public:
  BoundaryConditions(Realm& realm) : realm_(realm) {}
  ~BoundaryConditions() = default;

  void load(const YAML::Node& node);

  // ease of access methods to particular boundary condition
  BoundaryCondition* operator[](int i)
  {
    return boundaryConditionVector_[i].get();
  }

  Realm& realm_;
  BoundaryConditionVector boundaryConditionVector_;
};

} // namespace nalu
} // namespace Sierra

#endif
