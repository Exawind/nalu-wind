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

using FieldDefTypes = std::variant<FieldDefScalar, FieldDefVector>;
using FieldPointerTypes = std::variant<ScalarFieldType*, VectorFieldType*>;

} // namespace nalu
} // namespace sierra

#endif /* FIELDDEFINITIONS_H_ */
