// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <FieldRegistry.h>

namespace sierra {
namespace nalu {

// clang-format off
//std::map<std::string, FieldEntity> presetFields{ {"velocity", {stk::topology::NODE_RANK}}, };
// clang-format on

FieldRegistry::FieldRegistry()
  : fieldEntityMap_({
      {"velocity",
       std::make_unique<VectorFieldEntity>(stk::topology::NODE_RANK)},
    })
{
}

VectorFieldEntity*
FieldRegistry::get_field_entity(std::string name)
{
  return dynamic_cast<VectorFieldEntity*>(fieldEntityMap_.at(name).get());
}
} // namespace nalu
} // namespace sierra