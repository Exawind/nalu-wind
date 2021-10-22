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

// once we move to c++17 we can just use a map with std::any like
// https://raymii.org/s/articles/Store_multiple_types_in_a_single_stdmap_in_cpp_just_like_a_python_dict.html
// this will solve the type/map issues and simplify the code

class FieldManager
{
public:
  // probably want this to be a singleton
  FieldManager(
    stk::mesh::MetaData& meta, FieldStateLogic logic = FieldStateLogic());
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
