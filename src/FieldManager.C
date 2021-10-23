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

// clang-format off
#define REGISTER(TYPE, SIZE) auto* id = &(metaData_.declare_field<TYPE>( \
    def.rank, name, def.get_states(stateLogic_))); \
  for (auto&& p : parts) \
    stk::mesh::put_field_on_mesh(*id, *p, SIZE, nullptr); \
  break
// clang-format on

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
    REGISTER(VectorFieldType, 3);
  }
  case FieldTypes::SCALAR: {
    REGISTER(ScalarFieldType, 1);
  }
  case FieldTypes::GENERIC: {
    // TODO this size needs to be encoded in the FieldDefinition
    REGISTER(GenericFieldType, 1);
  }
  case FieldTypes::GLOBALID: {
    REGISTER(GlobalIdFieldType, 1);
  }
  case FieldTypes::LOCALID: {
    REGISTER(LocalIdFieldType, 1);
  }
  case FieldTypes::TPETID: {
    REGISTER(TpetIDFieldType, 1);
  }
  case FieldTypes::HYPREID: {
    REGISTER(HypreIDFieldType, 1);
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
  case FieldTypes::GENERIC: {
    return metaData_.get_field<GenericFieldType>(def.rank, name) != nullptr;
  }
  case FieldTypes::GLOBALID: {
    return metaData_.get_field<GlobalIdFieldType>(def.rank, name) != nullptr;
  }
  case FieldTypes::LOCALID: {
    return metaData_.get_field<LocalIdFieldType>(def.rank, name) != nullptr;
  }
  case FieldTypes::TPETID: {
    return metaData_.get_field<TpetIDFieldType>(def.rank, name) != nullptr;
  }
  case FieldTypes::HYPREID: {
    return metaData_.get_field<HypreIDFieldType>(def.rank, name) != nullptr;
  }
  default:
    throw std::runtime_error(
      "FieldTypes value not implemented in FieldManager");
    return false;
  }
}
#undef REGISTER
} // namespace nalu
} // namespace sierra