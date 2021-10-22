// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <FieldRegistry.h>
#include <FieldManager.h>
#include <stdexcept>

namespace sierra {
namespace nalu {

FieldManager::FieldManager(stk::mesh::MetaData& meta, FieldStateLogic logic)
  : metaData_(meta), stateLogic_(logic)
{
}

FieldDefinition
FieldManager::get_field_definition(std::string name)
{
  // right now this will throw if the field isn't registered in the map
  // maybe we'd want a compile time error though for developer mistakes?
  return FieldRegistry::query(name);
}

void
FieldManager::register_field(
  std::string name, const stk::mesh::PartVector& parts)
{
  const auto def = get_field_definition(name);
  // i'd like to remove this switch and just grab the type based off the enum
  // value but I haven't found a good way yet
  // I was considering something like this:
  // https://stackoverflow.com/questions/41415265/map-enum-value-to-a-type-in-c
  // but this only seems valid for concrete types
  switch (def.ftype) {
  case FieldTypes::VECTOR: {
    auto* id = &(metaData_.declare_field<VectorFieldType>(
      def.rank, name, def.get_states(stateLogic_)));
    for (auto&& p : parts)
      stk::mesh::put_field_on_mesh(*id, *p, nullptr);
    break;
  }
  case FieldTypes::SCALAR: {
    auto* id = &(metaData_.declare_field<ScalarFieldType>(
      def.rank, name, def.get_states(stateLogic_)));
    for (auto&& p : parts)
      stk::mesh::put_field_on_mesh(*id, *p, nullptr);
    break;
  }
  default:
    throw std::runtime_error(
      "FieldTypes value not implemented in FieldManager");
    break;
  }
}

bool
FieldManager::field_exists(std::string name)
{
  const auto def = get_field_definition(name);
  switch (def.ftype) {
  case FieldTypes::VECTOR: {
    return metaData_.get_field<VectorFieldType>(def.rank, name) != nullptr;
  }
  case FieldTypes::SCALAR: {
    return metaData_.get_field<ScalarFieldType>(def.rank, name) != nullptr;
  }
  default:
    throw std::runtime_error(
      "FieldTypes value not implemented in FieldManager");
    return false;
  }
}
} // namespace nalu
} // namespace sierra