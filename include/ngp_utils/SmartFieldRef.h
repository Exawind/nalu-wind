// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SMARTFIELDREF_H
#define SMARTFIELDREF_H
#include <Kokkos_Macros.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>

namespace tags {
// clang-format off

//ACCESS TYPES
struct READ{};
struct WRITE{};
struct READ_WRITE{};

// MEMSPACE
struct HOST{};
struct DEVICE{};
struct LEGACY{};

// clang-format on
} // namespace tags

namespace sierra::nalu {

using namespace tags;

template <
  typename FieldType,
  typename MEMSPACE,
  typename ACCESS,
  typename Enable = void>
class SmartFieldRef
{
};

template <typename FieldType, typename ACCESS>
class SmartFieldRef<
  FieldType,
  LEGACY,
  ACCESS,
  typename std::enable_if_t<
    std::is_base_of<stk::mesh::FieldBase, FieldType>::value>>
{
public:
  using T = typename FieldType::value_type;

  SmartFieldRef(FieldType& fieldRef) : fieldRef_(fieldRef) {}
  SmartFieldRef(const SmartFieldRef& src) : fieldRef_(src.fieldRef_)
  {
    if (is_read_) {
      fieldRef_.sync_to_host();
    } else {
      fieldRef_.clear_sync_state();
    }
  }
  // --- Default Accessors
  template <typename A = ACCESS>
  typename std::enable_if_t<!std::is_same<A, READ>::value, T>&
  get(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(fieldRef_, entity);
  }

  template <typename A = ACCESS>
  typename std::enable_if_t<!std::is_same<A, READ>::value, T>&
  operator()(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(fieldRef_, entity);
  }

  // --- Const Accessors
  template <typename A = ACCESS>
  const typename std::enable_if_t<std::is_same<A, READ>::value, T>&
  get(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(fieldRef_, entity);
  }

  template <typename A = ACCESS>
  const typename std::enable_if_t<std::is_same<A, READ>::value, T>&
  operator()(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(fieldRef_, entity);
  }

  ~SmartFieldRef()
  {
    if (is_write_) {
      // LEGACY implementation needs to be used in a limited scope.
      // Redundant usage is fine since sync's and mod's will be no-ops
      // but long lived cases like alg class members will negate the
      // purpose of this abstraction
      fieldRef_.modify_on_host();
    }
  }

private:
  static constexpr bool is_read_{
    std::is_same<ACCESS, READ>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  static constexpr bool is_write_{
    std::is_same<ACCESS, WRITE>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  FieldType& fieldRef_;
};

template <typename FieldType, typename MEMSPACE, typename ACCESS>
class SmartFieldRef<
  FieldType,
  MEMSPACE,
  ACCESS,
  typename std::enable_if_t<
    std::is_base_of<stk::mesh::NgpFieldBase, FieldType>::value>>
{
public:
  using T = typename FieldType::value_type;

  SmartFieldRef(FieldType fieldRef) : fieldRef_(fieldRef) {}

  SmartFieldRef(const SmartFieldRef& src)
    : fieldRef_(src.fieldRef_), is_copy_constructed_(true)
  {
    if (is_read_) {
      if (is_device_space) {
        fieldRef_.sync_to_device();
      } else {
        fieldRef_.sync_to_host();
      }
    } else {
      fieldRef_.clear_sync_state();
    }
  }

  ~SmartFieldRef()
  {
    if (is_write_) {
      if (is_copy_constructed_) {
        // NgpFieldBase implementations should only ever execute inside a
        // kokkos::paralle_for and hence be captured by a lambda. Therefore we
        // only ever need to sync copies that will have been snatched up through
        // lambda capture.
        fieldRef_.modify_on_device();
      } else {
        // try not requiring copy mechanism for host
        fieldRef_.modify_on_host();
      }
    }
  }

  //************************************************************
  // Host functions (Remove KOKKOS_FUNCTION decorators)
  //************************************************************
  template <typename M = MEMSPACE>
  std::enable_if_t<std::is_same<M, HOST>::value, unsigned> get_ordinal() const
  {
    return fieldRef_.get_ordinal();
  }

  // --- Default Accessors
  template <typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  template <typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return fieldRef_.operator()(index, component);
  }

  // --- Const Accessors
  template <typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  template <typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return fieldRef_.operator()(index, component);
  }
  //************************************************************
  // Device functions
  //************************************************************
  KOKKOS_FUNCTION
  template <typename M = MEMSPACE>
  std::enable_if_t<std::is_same<M, DEVICE>::value, unsigned> get_ordinal() const
  {
    return fieldRef_.get_ordinal();
  }

  // --- Default Accessors
  KOKKOS_FUNCTION
  template <typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  template <typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return fieldRef_.operator()(index, component);
  }

  // --- Const Accessors
  KOKKOS_FUNCTION
  template <typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  template <typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return fieldRef_.operator()(index, component);
  }

private:
  static constexpr bool is_device_space{std::is_same<MEMSPACE, DEVICE>::value};

  static constexpr bool is_read_{
    std::is_same<ACCESS, READ>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  static constexpr bool is_write_{
    std::is_same<ACCESS, WRITE>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  FieldType fieldRef_;
  const bool is_copy_constructed_{false};
};

template <typename MEMSPACE, typename ACCESS = READ_WRITE>
struct MakeFieldRef
{
  template <typename FieldType>
  SmartFieldRef<FieldType, MEMSPACE, ACCESS> operator()(FieldType& field)
  {
    return SmartFieldRef<FieldType, MEMSPACE, ACCESS>(field);
  }
};

} // namespace sierra::nalu

#endif
