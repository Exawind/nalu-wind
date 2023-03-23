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

#include "FieldRegistry.h"
#include <stk_mesh/base/FieldState.hpp>
#include <string>

namespace stk {
namespace mesh {
class MetaData;
}
} // namespace stk

namespace sierra {
namespace nalu {

class FieldManager
{
private:
  stk::mesh::MetaData& meta_;
  const int numStates_;
  const int numDimensions_;

public:
  FieldManager(stk::mesh::MetaData& meta, int numStates);

  FieldPointerTypes register_field(
    std::string name, const stk::mesh::PartVector& parts, int numStates = 0);

  FieldPointerTypes register_field(
    std::string name, const stk::mesh::Part& part, int numStates = 0);

  template <typename T>
  T get_field_ptr(
    std::string name,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone)
  {
    auto fieldDef = FieldRegistry::query(numDimensions_, numStates_, name);
    auto pointerSet = std::visit(
      [&](auto def) -> FieldPointerTypes {
        return &meta_
                  .get_field<typename decltype(def)::FieldType>(def.rank, name)
                  ->field_of_state(state);
      },
      fieldDef);
    return std::get<T>(pointerSet);
  }

  bool field_exists(std::string name);
};
} // namespace nalu
} // namespace sierra

#endif /* FIELDMANAGER_H_ */
