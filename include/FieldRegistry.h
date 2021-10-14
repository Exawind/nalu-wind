// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FIELDREGISTRY_H_
#define FIELDREGISTRY_H_

#include <string>
#include <map>
#include <memory>
#include <stk_topology/topology.hpp>
#include <FieldTypeDef.h>

namespace sierra {
namespace nalu {

// once we move to c++17 we can just use a map with std::any like
// https://raymii.org/s/articles/Store_multiple_types_in_a_single_stdmap_in_cpp_just_like_a_python_dict.html

class FieldEntityInterface
{
  virtual ~FieldEntityInterface() = default;
};

template <typename T>
class FieldEntity : public FieldEntityInterface
{
public:
  using type = T;
  FieldEntity<T>(stk::topology::rank_t rank) : rank(rank) {}
  FieldEntity<T>(FieldEntity<T>& entity) { return FieldEntity<T>(entity.rank); }
  stk::topology::rank_t rank;
};

using ScalarFieldEntity = FieldEntity<ScalarFieldType>;
using VectorFieldEntity = FieldEntity<VectorFieldType>;

class FieldRegistry
{
public:
  // probably want this to be a singleton
  FieldRegistry();
  VectorFieldEntity* get_field_entity(std::string name);
  // void register_field(stk::mesh::MetaData& meta, std::string)

private:
  const std::map<std::string, std::unique_ptr<FieldEntityInterface>>
    fieldEntityMap_;
};

} // namespace nalu
} // namespace sierra

#endif /* FIELDREGISTRY_H_ */
