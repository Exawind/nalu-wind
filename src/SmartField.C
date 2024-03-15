// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <SmartField.h>
#include <FieldTypeDef.h>

namespace sierra::nalu {
using namespace tags;

#define EXPLICIT_TYPE_INSTANTIATOR_NGP(T)                                      \
  template class SmartField<stk::mesh::NgpField<T>, DEVICE, READ>;             \
  template class SmartField<stk::mesh::NgpField<T>, DEVICE, WRITE_ALL>;        \
  template class SmartField<stk::mesh::NgpField<T>, DEVICE, READ_WRITE>;       \
  template class SmartField<stk::mesh::HostField<T>, HOST, READ>;              \
  template class SmartField<stk::mesh::HostField<T>, HOST, WRITE_ALL>;         \
  template class SmartField<stk::mesh::HostField<T>, HOST, READ_WRITE>

#define EXPLICIT_TYPE_INSTANTIATOR_LEGACY(T)                                   \
  template class SmartField<stk::mesh::Field<T>, LEGACY, READ>;                \
  template class SmartField<stk::mesh::Field<T>, LEGACY, WRITE_ALL>;           \
  template class SmartField<stk::mesh::Field<T>, LEGACY, READ_WRITE>

EXPLICIT_TYPE_INSTANTIATOR_NGP(int);
EXPLICIT_TYPE_INSTANTIATOR_NGP(double);
EXPLICIT_TYPE_INSTANTIATOR_NGP(stk::mesh::EntityId);

// Hypre Integer types
// What to do about HYPRE int vs long vs long long here?
/* #ifdef NALU_USES_HYPRE */
/* typedef HYPRE_Int HypreIntType; */
/* EXPLICIT_TYPE_INSTANTIATOR_NGP(HypreIntType); */
/* EXPLICIT_TYPE_INSTANTIATOR_LEGACY(HypreIDFieldType); */
/* #endif */

EXPLICIT_TYPE_INSTANTIATOR_LEGACY(int);
EXPLICIT_TYPE_INSTANTIATOR_LEGACY(double);
EXPLICIT_TYPE_INSTANTIATOR_LEGACY(stk::mesh::EntityId);
EXPLICIT_TYPE_INSTANTIATOR_LEGACY(LocalId);
EXPLICIT_TYPE_INSTANTIATOR_LEGACY(TpetIdType);

} // namespace sierra::nalu
