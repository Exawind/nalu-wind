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
#include "stk_io/IossBridge.hpp"

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
      return meta_.get_field<typename decltype(def)::DataType>(
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
  const void* initVal) const
{
  auto definition = FieldRegistry::query(numDimensions_, numStates_, name);

  return std::visit(
    [&](auto def) -> FieldPointerTypes {
      using val_type = typename decltype(def)::DataType;
      const int num_states = numStates ? numStates : def.num_states;
      const int num_components =
        numComponents ? numComponents : def.num_components;
      const FieldLayout layout = def.layout;

      const val_type* init = static_cast<const val_type*>(init_val);
      auto* id = &(meta_.declare_field<val_type>(def.rank, name, num_states));

      for (auto&& part : parts) {
        stk::mesh::put_field_on_mesh(*id, *part, num_components, init);

        if (layout == FieldLayout::VECTOR) {
          stk::io::set_field_output_type(
            *id, stk::io::FieldOutputType::VECTOR_3D);
        } else if (layout == FieldLayout::TENSOR) {
          stk::io::set_field_output_type(
            *id, stk::io::FieldOutputType::FULL_TENSOR_36);
        }
      }
#if 0
      std::cout << "Registring field '" << name << "' on parts:";
      for (const auto& part : parts)
        std::cout << " '" << part->name() << "'";
      std::cout << " with number of states " << num_states;
      std::cout << " and spatial dimension " << numDimensions_;
      std::cout << " with number of components " << num_components;
      std::cout << std::endl;
#endif
      return id;
    },
    definition);
}
} // namespace nalu
} // namespace sierra
