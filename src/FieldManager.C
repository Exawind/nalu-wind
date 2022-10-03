// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#include "FieldManager.h"
#include "stk_mesh/base/MetaData.hpp"

namespace sierra {
namespace nalu {

FieldManager::FieldManager(stk::mesh::MetaData& meta, int numStates)
  : meta_(meta), numStates_(numStates)
{
}

bool
FieldManager::field_exists(std::string name)
{
  auto definition = FieldRegistry::query(numStates_, name);

  return std::visit(
    [&](auto def) -> bool {
      return meta_.get_field<typename decltype(def)::FieldType>(
               def.rank, name) != nullptr;
    },
    definition);
}

FieldPointerTypes
FieldManager::register_field(
  std::string name, const stk::mesh::PartVector& parts)
{
  auto definition = FieldRegistry::query(numStates_, name);

  return std::visit(
    [&](auto def) -> FieldPointerTypes {
      auto* id = &(meta_.declare_field<typename decltype(def)::FieldType>(
        def.rank, name, def.num_states));

      for (auto&& part : parts) {
        stk::mesh::put_field_on_mesh(*id, *part, def.num_states, nullptr);
      }

      return id;
    },
    definition);
}

FieldPointerTypes
FieldManager::register_field(std::string name, const stk::mesh::Part& part)
{
  auto definition = FieldRegistry::query(numStates_, name);

  return std::visit(
    [&](auto def) -> FieldPointerTypes {
      auto* id = &(meta_.declare_field<typename decltype(def)::FieldType>(
        def.rank, name, def.num_states));

      stk::mesh::put_field_on_mesh(*id, part, def.num_states, nullptr);

      return id;
    },
    definition);
}
} // namespace nalu
} // namespace sierra
