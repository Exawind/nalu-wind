// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef FIELDMANAGER_H_
#define FIELDMANAGER_H_

#include "FieldRegistry.h"
#include "SmartField.h"
#include <stk_mesh/base/FieldState.hpp>
#include "stk_mesh/base/GetNgpField.hpp"
#include <string>
#include <type_traits>

namespace stk::mesh {
class MetaData;
} // namespace stk::mesh

namespace sierra::nalu {

class FieldManager
{
private:
  stk::mesh::MetaData& meta_;
  const int numStates_;
  const int numDimensions_;

public:
  /// The FieldManager will be created with the number of
  /// spatial dimensions as defined in the MetaData and the
  /// number of states as passed.  The fields will be registered
  /// based on these two keys.  Another FieldManager with the
  /// same spatial dimension and number of states will share
  /// the same field Registry and hence the same defined fields.
  /// A different number of spatial dimensions or states and
  /// a different field Registry will be used and the list of
  /// registered fields will be independent.
  FieldManager(stk::mesh::MetaData& meta, const int numStates);

  /// Register a field using the definition in the field Registry.
  /// The number of states and the dimension of the field will
  /// be taken from the field Registry class. The type of the field
  /// in the template parameter can be deduced from the field type
  /// as specified in the FieldRegistry.C class.
  ///
  /// The "velocity" field is of type MultiStateNodalVector in the
  /// field Registry class, so the field is of type VectorFieldType:
  /// VectorFieldType* velocity = fieldManager->register_field<VectorFieldType>(
  ///    "velocity", universal_mesh_part);
  ///
  /// A field of type SingleStateElemScalar would be assigned to
  /// ScalarFieldType* and so on.
  ///
  template <typename T>
  T* register_field(
    const std::string& name,
    const stk::mesh::PartVector& parts,
    const void* init_val = nullptr,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone) const
  {
    const int numStates = 0;
    const int numComponents = 0;
    register_field(name, parts, numStates, numComponents, init_val);
    return get_field_ptr<T>(name, state);
  }

  /// Check to see if the field has been registered.
  bool field_exists(const std::string& name) const;

  unsigned size() const { return meta_.get_fields().size(); }
  /// Register a Generic field.
  /// A Generic field is of type: SingleStateElemGeneric,
  /// SingleStateEdgeGeneric, SingleStateNodeGeneric,... For a generic field the
  /// numStates and numComponents should be specified.
  GenericFieldType* register_generic_field(
    const std::string& name,
    const stk::mesh::PartVector& parts,
    const int numStates,
    const int numComponents,
    const void* init_val = nullptr,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone) const
  {
    register_field(name, parts, numStates, numComponents, init_val);
    return get_field_ptr<GenericFieldType>(name, state);
  }

  // Return a field by the given name and of type T, the template parameter.
  // This function will throw if the named field is not of the type
  // specified by the template parameter: ScalarFieldType, VectorFieldType,
  // ScalarIntFieldType, GlobalIdFieldType,....
  template <typename T>
  T* get_field_ptr(
    const std::string& name,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone) const
  {
    FieldDefTypes fieldDef =
      FieldRegistry::query(numDimensions_, numStates_, name);
    FieldPointerTypes pointerSet = std::visit(
      [&](auto def) -> FieldPointerTypes {
        return &meta_
                  .get_field<typename decltype(def)::FieldType>(def.rank, name)
                  ->field_of_state(state);
      },
      fieldDef);
    return std::get<T*>(pointerSet);
  }

  /// Register a field with the option to override default parameters that
  /// would otherwise be defined in the field Registry class.
  ///
  /// If numStates = 0 then the number of states comes from the
  /// field Registry.  Same for numComponents = 0 and init_val = nullptr.
  ///
  /// This is useful for dynamic fields that depend on the input
  /// options to define the number of states or number of components since the
  /// field Registry is a static compile-time definition. Care must be taken
  /// not to re-register the same field on the same parts with a conflicting
  /// number of states or components.
  FieldPointerTypes register_field (
    const std::string& name,
    const stk::mesh::PartVector& parts,
    const int numStates = 0,
    const int numComponents = 0,
    const void* init_val = nullptr) const;

  /// Given the named field that has already been registered on the CPU
  /// return the GPU version of the same field.
  template <typename T>
  stk::mesh::NgpField<T>& get_ngp_field_ptr(
    std::string name,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone) const
  {
    FieldDefTypes fieldDef =
      FieldRegistry::query(numDimensions_, numStates_, name);
    const stk::mesh::FieldBase& stkField = std::visit(
      [&](auto def) -> stk::mesh::FieldBase& {
        return meta_
          .get_field<typename decltype(def)::FieldType>(def.rank, name)
          ->field_of_state(state);
      },
      fieldDef);
    stk::mesh::NgpField<T>& tmp = stk::mesh::get_updated_ngp_field<T>(stkField);
    return tmp;
  }

  template <typename T, typename ACCESS>
  SmartField<stk::mesh::NgpField<T>, tags::DEVICE, ACCESS>
  get_device_smart_field(
    std::string name,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone) const
  {
    return MakeSmartField<tags::DEVICE, ACCESS>().template operator()<T>(
      get_ngp_field_ptr<T>(name, state));
  }

  template <typename T, typename ACCESS>
  SmartField<T, tags::LEGACY, ACCESS> get_legacy_smart_field(
    std::string name,
    stk::mesh::FieldState state = stk::mesh::FieldState::StateNone) const
  {
    return MakeSmartField<tags::LEGACY, ACCESS>().template operator()<T>(
      get_field_ptr<T>(name, state));
  }
};
} // namespace sierra::nalu

#endif /* FIELDMANAGER_H_ */
