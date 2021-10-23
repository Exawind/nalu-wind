// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FIELDMANAGER_H_
#define FIELDMANAGER_H_

#include <string>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <stk_topology/topology.hpp>
#include <FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
using PartVector = std::vector<Part*>;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class FieldManager
{
public:
  FieldManager(stk::mesh::MetaData& meta, FieldStateLogic logic = {false});
  FieldDefinition get_field_definition(std::string name);
  void register_field(std::string name, const stk::mesh::PartVector& parts);
  bool field_exists(std::string name);

private:
  stk::mesh::MetaData& metaData_;
  const FieldStateLogic stateLogic_;
};

} // namespace nalu
} // namespace sierra

#endif /* FIELDMANAGER_H_ */
