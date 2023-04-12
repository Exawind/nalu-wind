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

FieldManager::FieldManager(stk::mesh::MetaData& meta, const int numStates)
  : meta_(meta), numStates_(numStates), numDimensions_(meta.spatial_dimension())
{
}

bool
FieldManager::field_exists(const std::string& name)
{
  auto definition = FieldRegistry::query(numDimensions_, numStates_, name);

  return std::visit(
    [&](auto def) -> bool {
      return meta_.get_field<typename decltype(def)::FieldType>(
               def.rank, name) != nullptr;
    },
    definition);
}

FieldPointerTypes
FieldManager::register_field(
  const std::string& name,
  const stk::mesh::PartVector& parts,
  const int numStates,
  const int numComponents,
  const void* init_val)
{
  auto definition = FieldRegistry::query(numDimensions_, numStates_, name);

  return std::visit(
    [&](auto def) -> FieldPointerTypes {
      using field_type = typename decltype(def)::FieldType;
      using val_type = typename stk::mesh::FieldTraits<field_type>::data_type;
      const int num_states = numStates ? numStates : def.num_states;
      const int num_components =
        numComponents ? numComponents : def.num_components;

      const val_type* init = static_cast<const val_type*>(init_val);
      auto* id = &(meta_.declare_field<field_type>(def.rank, name, num_states));
      for (auto&& part : parts) {
        stk::mesh::put_field_on_mesh(*id, *part, num_components, init);
      }

      return id;
    },
    definition);
}
} // namespace nalu
} // namespace sierra
