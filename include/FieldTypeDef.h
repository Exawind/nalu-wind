// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FieldTypeDef_h
#define FieldTypeDef_h

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>

#ifdef NALU_USES_TRILINOS_SOLVERS
#include <Tpetra_Details_DefaultTypes.hpp>
#endif
#include <functional>

#ifdef NALU_USES_HYPRE
#include "HYPRE_utilities.h"
#endif

namespace sierra {
namespace nalu {

using ScalarFieldType = stk::mesh::Field<double>;
using GlobalIdFieldType = stk::mesh::Field<stk::mesh::EntityId>;
using ScalarIntFieldType = stk::mesh::Field<int>;
using NGPDoubleFieldType = stk::mesh::NgpField<double>;
using NGPGlobalIdFieldType = stk::mesh::NgpField<stk::mesh::EntityId>;
using NGPScalarIntFieldType = stk::mesh::NgpField<int>;

using VectorFieldType = stk::mesh::Field<double>;
using TensorFieldType = stk::mesh::Field<double>;

using GenericFieldType = stk::mesh::Field<double>;

using GenericIntFieldType = stk::mesh::Field<int>;

using LocalId = unsigned;
using LocalIdFieldType = stk::mesh::Field<LocalId>;

#ifdef NALU_USES_HYPRE
using HypreIntType = HYPRE_Int;
#else
using HypreIntType = int;
#endif

#ifdef NALU_USES_TRILINOS_SOLVERS
using TpetIdType = Tpetra::Details::DefaultTypes::global_ordinal_type;
#else
using TpetIdType = int64_t;
#endif

using TpetIDFieldType = stk::mesh::Field<TpetIdType>;
using HypreIDFieldType = stk::mesh::Field<HypreIntType>;
using NGPHypreIDFieldType = stk::mesh::NgpField<HypreIntType>;

} // namespace nalu
} // namespace sierra

#endif
