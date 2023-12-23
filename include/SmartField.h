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
template <typename FieldType, typename MEMSPACE, typename ACCESS>
class SmartField
{
};

// LEGACY implementation, HOST only, data type has to be a reference b/c the
// stk::mesh::field ctor is not public
//
// This Type should be used as close to a bucket loop as possible, and not
// stored as a class member since sync/modify are marked in the ctor/dtor
template <typename FieldType, typename ACCESS>
class SmartField<FieldType, tags::LEGACY, ACCESS>
{

private:
  static constexpr bool is_read_{
    std::is_same_v<ACCESS, READ> || std::is_same_v<ACCESS, READ_WRITE>};

  static constexpr bool is_write_{
    std::is_same_v<ACCESS, WRITE_ALL> || std::is_same_v<ACCESS, READ_WRITE>};

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
  inline typename std::enable_if_t<!std::is_same_v<A, READ>, T>*
  get(const stk::mesh::Entity& entity) const
  {
    return stk::mesh::field_data(stkField_, entity);
  }

  template <typename A = ACCESS>
  inline typename std::enable_if_t<!std::is_same_v<A, READ>, T>*
  operator()(const stk::mesh::Entity& entity) const
  {
    return stk::mesh::field_data(stkField_, entity);
  }

  // --- Const Accessors
  template <typename A = ACCESS>
  inline const typename std::enable_if_t<std::is_same_v<A, READ>, T>*
  get(const stk::mesh::Entity& entity) const
  {
    return stk::mesh::field_data(stkField_, entity);
  }

  template <typename A = ACCESS>
  inline const typename std::enable_if_t<std::is_same_v<A, READ>, T>*
  operator()(const stk::mesh::Entity& entity) const
  {
    return stk::mesh::field_data(stkField_, entity);
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

// DEVICE
//
// These should always be used as part of lambda/functor captures
// using copy by value.
//
template <typename FieldType, typename ACCESS>
class SmartField<FieldType, tags::DEVICE, ACCESS>
{
private:
  static constexpr bool is_read_{
    std::is_same_v<ACCESS, READ> || std::is_same_v<ACCESS, READ_WRITE>};

  static constexpr bool is_write_{
    std::is_same_v<ACCESS, WRITE_ALL> || std::is_same_v<ACCESS, READ_WRITE>};

  FieldType stkField_;
  const bool is_copy_constructed_{false};

public:
  using T = typename FieldType::value_type;

  KOKKOS_FUNCTION
  SmartField(FieldType fieldRef) : stkField_(fieldRef) {}

  KOKKOS_FUNCTION
  SmartField(const SmartField& src)
    : stkField_(src.stkField_), is_copy_constructed_(true)
  {
    KOKKOS_IF_ON_HOST(
      if (is_read_) { stkField_.sync_to_device(); } else {
        stkField_.clear_sync_state();
      });
  }

  KOKKOS_FUNCTION
  ~SmartField()
  {
    KOKKOS_IF_ON_HOST(if (is_write_ && is_copy_constructed_) {
      // NgpFieldBase implementations should only ever execute inside a
      // kokkos::paralle_for and hence be captured by a lambda. Therefore we
      // only ever need to sync copies that will have been snatched up through
      // lambda capture.
      stkField_.modify_on_device();
    });
  }

  //************************************************************
  // Device functions
  //************************************************************
  KOKKOS_FORCEINLINE_FUNCTION
  unsigned get_ordinal() const { return stkField_.get_ordinal(); }

  // --- Default Accessors
  template <typename Mesh, typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_same_v<A, READ>, T>&
  get(const Mesh& ngpMesh, stk::mesh::Entity entity, int component) const
  {
    return stkField_.get(ngpMesh, entity, component);
  }

  template <typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_same_v<A, READ>, T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_same_v<A, READ>, T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_same_v<A, READ>, T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_same_v<A, READ>, T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }

  // --- Const Accessors
  template <typename Mesh, typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION const
    std::enable_if_t<std::is_same_v<A, READ>, T>&
    get(const Mesh& ngpMesh, stk::mesh::Entity entity, int component) const
  {
    return stkField_.get(ngpMesh, entity, component);
  }

  template <typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION const
    std::enable_if_t<std::is_same_v<A, READ>, T>&
    get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION const
    std::enable_if_t<std::is_same_v<A, READ>, T>&
    get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION const
    std::enable_if_t<std::is_same_v<A, READ>, T>&
    operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  KOKKOS_FORCEINLINE_FUNCTION const
    std::enable_if_t<std::is_same_v<A, READ>, T>&
    operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }
};

// HOST implementations
//
// These should always be used as part of lambda/functor captures
// using copy by value.
//
template <typename FieldType, typename ACCESS>
class SmartField<FieldType, tags::HOST, ACCESS>
{
private:
  static constexpr bool is_read_{
    std::is_same_v<ACCESS, READ> || std::is_same_v<ACCESS, READ_WRITE>};

  static constexpr bool is_write_{
    std::is_same_v<ACCESS, WRITE_ALL> || std::is_same_v<ACCESS, READ_WRITE>};

  FieldType stkField_;
  const bool is_copy_constructed_{false};

public:
  using T = typename FieldType::value_type;

  SmartField(FieldType fieldRef) : stkField_(fieldRef) {}

  SmartField(const SmartField& src)
    : stkField_(src.stkField_), is_copy_constructed_(true)
  {
    if (is_read_) {
      stkField_.sync_to_host();
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
        stkField_.modify_on_host();
      }
    }
  }

  //************************************************************
  // Host functions (Remove KOKKOS_FORCEINLINE_FUNCTION decorators)
  //************************************************************
  inline unsigned get_ordinal() const { return stkField_.get_ordinal(); }

  // --- Default Accessors
  template <typename Mesh, typename A = ACCESS>
  inline std::enable_if_t<!std::is_same_v<A, READ>, T>&
  get(const Mesh& ngpMesh, stk::mesh::Entity entity, int component) const
  {
    return stkField_.get(ngpMesh, entity, component);
  }

  template <typename A = ACCESS>
  inline std::enable_if_t<!std::is_same_v<A, READ>, T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  inline std::enable_if_t<!std::is_same_v<A, READ>, T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS>
  inline std::enable_if_t<!std::is_same_v<A, READ>, T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  inline std::enable_if_t<!std::is_same_v<A, READ>, T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }

  // --- Const Accessors
  template <typename Mesh, typename A = ACCESS>
  inline const std::enable_if_t<std::is_same_v<A, READ>, T>&
  get(const Mesh& ngpMesh, stk::mesh::Entity entity, int component) const
  {
    return stkField_.get(ngpMesh, entity, component);
  }

  template <typename A = ACCESS>
  inline const std::enable_if_t<std::is_same_v<A, READ>, T>&
  get(stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  inline const std::enable_if_t<std::is_same_v<A, READ>, T>&
  get(MeshIndex index, int component) const
  {
    return stkField_.get(index, component);
  }

  template <typename A = ACCESS>
  inline const std::enable_if_t<std::is_same_v<A, READ>, T>&
  operator()(const stk::mesh::FastMeshIndex& index, int component) const
  {
    return stkField_(index, component);
  }

  template <typename MeshIndex, typename A = ACCESS>
  inline const std::enable_if_t<std::is_same_v<A, READ>, T>&
  operator()(const MeshIndex index, int component) const
  {
    return stkField_(index, component);
  }
};

template <typename MEMSPACE, typename ACCESS = READ_WRITE>
struct MakeSmartField
{
};

template <typename ACCESS>
struct MakeSmartField<LEGACY, ACCESS>
{
  // use pointer since that is the common access type for stk::mesh::Field<T>
  template <typename T>
  SmartField<stk::mesh::Field<T>, LEGACY, ACCESS>
  operator()(stk::mesh::Field<T>* field)
  {
    return SmartField<stk::mesh::Field<T>, LEGACY, ACCESS>(*field);
  }
};

template <typename ACCESS>
struct MakeSmartField<HOST, ACCESS>
{
  template <typename T>
  SmartField<stk::mesh::HostField<T>, HOST, ACCESS>
  operator()(stk::mesh::HostField<T>& field)
  {
    return SmartField<stk::mesh::HostField<T>, HOST, ACCESS>(field);
  }
};

template <typename ACCESS>
struct MakeSmartField<DEVICE, ACCESS>
{
  template <typename T>
  SmartField<stk::mesh::NgpField<T>, DEVICE, ACCESS>
  operator()(stk::mesh::NgpField<T>& field)
  {
    return SmartField<stk::mesh::NgpField<T>, DEVICE, ACCESS>(field);
  }
};

template <typename T, typename ACCESS>
using SmartDeviceField = SmartField<T, DEVICE, ACCESS>;
template <typename T, typename ACCESS>
using SmartHostField = SmartField<T, HOST, ACCESS>;
template <typename T, typename ACCESS>
using SmartLegacyField = SmartField<T, LEGACY, ACCESS>;
} // namespace sierra::nalu

#endif
