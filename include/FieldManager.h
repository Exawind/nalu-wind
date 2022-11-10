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

public:
  FieldManager(stk::mesh::MetaData& meta, int numStates);

  FieldPointerTypes
  register_field(std::string name, const stk::mesh::PartVector& parts);

  FieldPointerTypes
  register_field(std::string name, const stk::mesh::Part& part);

  FieldPointerTypes get_field_ptr(std::string name)
  {
    auto fieldDef = FieldRegistry::query(numStates_, name);
    return std::visit(
      [&](auto def) -> FieldPointerTypes {
        return meta_.get_field<typename decltype(def)::FieldType>(
          def.rank, name);
      },
      fieldDef);
  }

  bool field_exists(std::string name);
};
} // namespace nalu
} // namespace sierra

#endif /* FIELDMANAGER_H_ */
