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

template <typename T>
struct FieldDefinition
{
  using FieldType = T;
  const stk::topology::rank_t rank;
  const int num_states{1};
};

using FieldDefScalar = FieldDefinition<ScalarFieldType>;
using FieldDefVector = FieldDefinition<VectorFieldType>;
using FieldDefGeneric = FieldDefinition<GenericFieldType>;
using FieldDefGenericInt = FieldDefinition<GenericIntFieldType>;
using FieldDefTpetraId = FieldDefinition<TpetIDFieldType>;
using FieldDefLocalId = FieldDefinition<LocalIdFieldType>;
using FieldDefGlobalId = FieldDefinition<GlobalIdFieldType>;
using FieldDefHypreId = FieldDefinition<HypreIDFieldType>;
using FieldDefScalarInt = FieldDefinition<ScalarIntFieldType>;

// Type redundancy can occur between HypreId and ScalarInt
// which will break std::variant
using FieldDefTypes = std::conditional<
  std::is_same_v<ScalarIntFieldType, HypreIDFieldType>,
  std::variant<
    FieldDefScalar,
    FieldDefVector,
    FieldDefGeneric,
    FieldDefGenericInt,
    FieldDefTpetraId,
    FieldDefLocalId,
    FieldDefGlobalId,
    FieldDefScalarInt>,
  std::variant<
    FieldDefScalar,
    FieldDefVector,
    FieldDefGeneric,
    FieldDefGenericInt,
    FieldDefTpetraId,
    FieldDefLocalId,
    FieldDefGlobalId,
    FieldDefScalarInt,
    FieldDefHypreId>>::type;

using FieldPointerTypes = std::conditional<
  std::is_same_v<ScalarIntFieldType, HypreIDFieldType>,
  std::variant<
    ScalarFieldType*,
    VectorFieldType*,
    GenericFieldType*,
    GenericIntFieldType*,
    TpetIDFieldType*,
    LocalIdFieldType*,
    GlobalIdFieldType*,
    ScalarIntFieldType*>,
  std::variant<
    ScalarFieldType*,
    VectorFieldType*,
    GenericFieldType*,
    GenericIntFieldType*,
    TpetIDFieldType*,
    LocalIdFieldType*,
    GlobalIdFieldType*,
    ScalarIntFieldType*,
    HypreIDFieldType*>>::type;

} // namespace nalu
} // namespace sierra

#endif /* FIELDDEFINITIONS_H_ */
