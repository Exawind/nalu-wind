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
namespace nalu {

class BoundaryCondition {
 public:
   BoundaryCondition() {}
   virtual ~BoundaryCondition() {}

   std::string bcName_;
   std::string targetName_;
   BoundaryConditionType theBcType_;
};

typedef std::vector<std::unique_ptr<BoundaryCondition>> BoundaryConditionVector;
struct BoundaryConditionCreator
{
public:
  BoundaryConditionVector create_bc_vector(const YAML::Node& node);

  std::unique_ptr<BoundaryCondition>
  load_single_bc_node(const YAML::Node& node);
};
} // namespace nalu
} // namespace Sierra

#endif
