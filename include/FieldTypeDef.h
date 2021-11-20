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
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>

#include <Tpetra_Details_DefaultTypes.hpp>
#include <functional>

#ifdef NALU_USES_HYPRE
#include "HYPRE_utilities.h"
#endif

namespace sierra {
namespace nalu {

// define scalar field typedef
typedef stk::mesh::Field<double> ScalarFieldType;
typedef stk::mesh::Field<stk::mesh::EntityId> GlobalIdFieldType;
typedef stk::mesh::Field<int> ScalarIntFieldType;
typedef stk::mesh::NgpField<double> NGPDoubleFieldType;
typedef stk::mesh::NgpField<stk::mesh::EntityId> NGPGlobalIdFieldType;
typedef stk::mesh::NgpField<int> NGPScalarIntFieldType;

// define vector field typedef; however, what is the value of Cartesian?
typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;

// define generic
typedef stk::mesh::Field<double, stk::mesh::SimpleArrayTag> GenericFieldType;

// field type for local ids
typedef unsigned LocalId;
typedef stk::mesh::Field<LocalId> LocalIdFieldType;

// Hypre Integer types
#ifdef NALU_USES_HYPRE
typedef HYPRE_Int HypreIntType;
#else
typedef int HypreIntType;
#endif

typedef stk::mesh::Field<Tpetra::Details::DefaultTypes::global_ordinal_type>
  TpetIDFieldType;
typedef stk::mesh::Field<HypreIntType> HypreIDFieldType;
typedef stk::mesh::NgpField<HypreIntType> NGPHypreIDFieldType;

} // namespace nalu
} // namespace sierra

#endif
