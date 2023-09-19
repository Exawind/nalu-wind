// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef SMARTFIELD_H
#define SMARTFIELD_H
#include <Kokkos_Macros.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>

namespace tags {
// clang-format off

//ACCESS TYPES
struct READ{};
struct WRITE_ALL{};
struct READ_WRITE{};

// MEMSPACE
struct HOST{};
struct DEVICE{};
struct LEGACY{};

// clang-format on
} // namespace tags

namespace sierra::nalu {

using namespace tags;
/* SmartField is a type that is designed to automatically handle field syncs
 * and modifies
 * It adheres to the pattern of sync before you use, and modify after
 *
 * There are 3 Template params:
 * - FieldType: Data type for the underlying stk field
 *
 * - MEMSPACE: where the field is valid
 *     - DEVICE: is self explanatory
 *     - HOST:   is using the same modern stk syntax based on
 *               kokkos::parallel_for but for host data
 *     - LEGACY: for the stk::mesh::field_data type access (bucket loops)
 *
 * - ACCESS: how the data can/should be used
 *     - READ:        read only/const data accessors (only syncs data)
 *     - READ_WRITE:  read and write data to field (syncs then marks modified)
 *     - WRITE_ALL:   should only be used when overwritting all the field_data
 *                    (clears sync state)
 *
 * NOTE: this implementation makes heavy use of SFINAE
 */
template <
  typename FieldType,
  typename MEMSPACE,
  typename ACCESS,
  typename Enable = void>
class SmartField
{
};

// LEGACY implementation, HOST only, data type has to be a reference b/c the 
// stk::mesh::field ctor is not public
//
// This Type should be used as close to a bucket loop as possible, and not 
// stored as a class member since sync/modify are marked in the ctor/dtor
template <typename FieldType, typename ACCESS>
class SmartField<
  FieldType,
  LEGACY,
  ACCESS,
  typename std::enable_if_t<
    std::is_base_of<stk::mesh::FieldBase, FieldType>::value>>
{

private:
  static constexpr bool is_read_{
    std::is_same<ACCESS, READ>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  static constexpr bool is_write_{
    std::is_same<ACCESS, WRITE_ALL>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  FieldType& stkField_;

public:
  using T = typename FieldType::value_type;

  SmartField(FieldType& fieldRef) : stkField_(fieldRef)
  {
    if (is_read_)
      stkField_.sync_to_host();
    else
      stkField_.clear_sync_state();
  }

  SmartField(const SmartField& src) : stkField_(src.stkField_)
  {
    if (is_read_)
      stkField_.sync_to_host();
    else
      stkField_.clear_sync_state();
  }

  // --- Default Accessors
  template <typename A = ACCESS>
  typename std::enable_if_t<!std::is_same<A, READ>::value, T>&
  get(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(stkField_, entity);
  }

  template <typename A = ACCESS>
  typename std::enable_if_t<!std::is_same<A, READ>::value, T>&
  operator()(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(stkField_, entity);
  }

  // --- Const Accessors
  template <typename A = ACCESS>
  const typename std::enable_if_t<std::is_same<A, READ>::value, T>&
  get(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(stkField_, entity);
  }

  template <typename A = ACCESS>
  const typename std::enable_if_t<std::is_same<A, READ>::value, T>&
  operator()(const stk::mesh::Entity& entity) const
  {
    return *stk::mesh::field_data(stkField_, entity);
  }

  ~SmartField()
  {
    if (is_write_) {
      // LEGACY implementation needs to be used in a limited scope.
      // Redundant usage is fine since sync's and mod's will be no-ops
      // but long lived cases like alg class members will negate the
      // purpose of this abstraction
      stkField_.modify_on_host();
    }
  }
};

// DEVICE and HOST implementations
//
// These should always be used as part of lambda/functor captures
// using copy by value.
//
// SFINAE is used to remove KOKKOS_FUNCTION type decorators for HOST MEMSPACE
template <typename FieldType, typename MEMSPACE, typename ACCESS>
class SmartField<
  FieldType,
  MEMSPACE,
  ACCESS,
  typename std::enable_if_t<
    std::is_base_of<stk::mesh::NgpFieldBase, FieldType>::value>>
{
private:
  static constexpr bool is_device_space_{std::is_same<MEMSPACE, DEVICE>::value};

  static constexpr bool is_read_{
    std::is_same<ACCESS, READ>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  static constexpr bool is_write_{
    std::is_same<ACCESS, WRITE_ALL>::value ||
    std::is_same<ACCESS, READ_WRITE>::value};

  FieldType stkField_;
  const bool is_copy_constructed_{false};

public:
  using T = typename FieldType::value_type;

  SmartField(FieldType fieldRef) : stkField_(fieldRef) {}

  SmartField(const SmartField& src)
    : stkField_(src.stkField_), is_copy_constructed_(true)
  {
    if (is_read_) {
      if (is_device_space_) {
        stkField_.sync_to_device();
      } else {
        stkField_.sync_to_host();
      }
    } else {
      stkField_.clear_sync_state();
    }
  }

  ~SmartField()
  {
    if (is_write_) {
      if (is_copy_constructed_) {
        // NgpFieldBase implementations should only ever execute inside a
        // kokkos::paralle_for and hence be captured by a lambda. Therefore we
        // only ever need to sync copies that will have been snatched up through
        // lambda capture.
        if (is_device_space_)
          stkField_.modify_on_device();
        else
          stkField_.modify_on_host();
      }
    }
  }

  //************************************************************
  // Host functions (Remove KOKKOS_FUNCTION decorators)
  //************************************************************
  template <typename M = MEMSPACE>
  std::enable_if_t<std::is_same<M, HOST>::value, unsigned> get_ordinal() const
  {
    return stkField_.get_ordinal();
  }

  // --- Default Accessors
  template <typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  std::enable_if_t<
    std::is_same<M, HOST>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }

  // --- Const Accessors
  template <typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  const std::enable_if_t<
    std::is_same<M, HOST>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }

  //************************************************************
  // Device functions
  //************************************************************
  template <typename M = MEMSPACE>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<M, DEVICE>::value, unsigned>
  get_ordinal() const
  {
    return stkField_.get_ordinal();
  }

  // --- Default Accessors
  template <typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION std::enable_if_t<
    std::is_same<M, DEVICE>::value && !std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }

  // --- Const Accessors
  template <typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS, typename M = MEMSPACE>
  KOKKOS_FUNCTION const std::enable_if_t<
    std::is_same<M, DEVICE>::value && std::is_same<A, READ>::value,
    T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }
};

template <typename MEMSPACE, typename ACCESS = READ_WRITE>
struct MakeSmartField
{
  template <typename FieldType>
  SmartField<FieldType, MEMSPACE, ACCESS> operator()(FieldType& field)
  {
    return SmartField<FieldType, MEMSPACE, ACCESS>(field);
  }
};

} // namespace sierra::nalu

#endif
