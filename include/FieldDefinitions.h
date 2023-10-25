// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//
#ifndef FIELDDEFINITIONS_H_
#define FIELDDEFINITIONS_H_
#include "FieldTypeDef.h"
#include <type_traits>
#include <variant>

namespace sierra {
namespace nalu {

enum class FieldLayout { SCALAR, VECTOR, TENSOR, ARRAY };

template <typename T, FieldLayout Layout = FieldLayout::SCALAR>
struct FieldDefinition
{
  using DataType = T;
  const stk::topology::rank_t rank;
  const int num_states{1};
  const int num_components{1};
  const FieldLayout layout{Layout};
};

using FieldDefScalar = FieldDefinition<double>;
using FieldDefVector = FieldDefinition<double, FieldLayout::VECTOR>;
using FieldDefTensor = FieldDefinition<double, FieldLayout::TENSOR>;
using FieldDefGeneric = FieldDefinition<double, FieldLayout::ARRAY>;
using FieldDefGenericInt = FieldDefinition<int, FieldLayout::ARRAY>;
using FieldDefTpetraId = FieldDefinition<TpetIdType>;
using FieldDefLocalId = FieldDefinition<LocalId>;
using FieldDefGlobalId = FieldDefinition<stk::mesh::EntityId>;
using FieldDefHypreId = FieldDefinition<HypreIntType>;
using FieldDefScalarInt = FieldDefinition<int>;

// Type redundancy can occur between HypreId and ScalarInt
// which will break std::variant
using FieldDefTypes = std::conditional<
  std::is_same_v<int, HypreIntType>,
  std::variant<
    FieldDefScalar,
    FieldDefVector,
    FieldDefTensor,
    FieldDefGeneric,
    FieldDefGenericInt,
    FieldDefTpetraId,
    FieldDefLocalId,
    FieldDefGlobalId,
    FieldDefScalarInt>,
  std::variant<
    FieldDefScalar,
    FieldDefVector,
    FieldDefTensor,
    FieldDefGeneric,
    FieldDefGenericInt,
    FieldDefTpetraId,
    FieldDefLocalId,
    FieldDefGlobalId,
    FieldDefScalarInt,
    FieldDefHypreId>>::type;

// Trouble!
using FieldPointerTypes = std::conditional<
  std::is_same_v<int, HypreIntType>,
  std::variant<
    stk::mesh::Field<double>*,
    stk::mesh::Field<int>*,
    stk::mesh::Field<LocalId>*,
    stk::mesh::Field<stk::mesh::EntityId>*,
    stk::mesh::Field<TpetIdType>*>,
  std::variant<
    stk::mesh::Field<double>*,
    stk::mesh::Field<int>*,
    stk::mesh::Field<LocalId>*,
    stk::mesh::Field<stk::mesh::EntityId>*,
    stk::mesh::Field<TpetIdType>*,
    stk::mesh::Field<HypreIntType>*>>::type;

} // namespace nalu
} // namespace sierra

#endif /* FIELDDEFINITIONS_H_ */
