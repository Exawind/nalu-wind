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

namespace sierra::nalu{

struct READ{};
struct WRITE{};
struct READ_WRITE{};

struct HOST{};
struct DEVICE{};

template <typename MEMSPACE, typename SCOPE, typename T>
class SmartFieldRef{};

template <typename SCOPE, typename T>
class SmartFieldRef<DEVICE, SCOPE, T>{
public:
  SmartFieldRef(stk::mesh::NgpField<T>& ngpField):
    fieldRef_(ngpField){}

  SmartFieldRef(const SmartFieldRef& src):
    fieldRef_(src.fieldRef_),
    is_copy_constructed_(true)
  {
    if(is_read())
      fieldRef_.sync_to_device();
    else
      fieldRef_.clear_sync_state();
  }

  // device implementations should only ever execute inside a kokkos::paralle_for
  // and hence be captured by a lambda.
  // Therefore we only ever need to sync copies that will have been snatched up
  // through lambda capture.
  ~SmartFieldRef(){
    if(is_copy_constructed_ && is_write()){
      fieldRef_.modify_on_device();
    }
  }

  KOKKOS_FUNCTION
  unsigned get_ordinal() const{
    return fieldRef_.get_ordinal();
  }

  // TODO make it so these accessors are read only for read type i.e. const correct
  // and give clear compile or runtime error for programming mistakes
  KOKKOS_FUNCTION
  T& get(stk::mesh::FastMeshIndex index, int component){
    return fieldRef_.get(index, component);
  }

  template<typename MeshIndex> KOKKOS_FUNCTION
  T& get(MeshIndex index, int component){
    return fieldRef_.get(index, component);
  }

  KOKKOS_FUNCTION
  T& operator()(stk::mesh::FastMeshIndex index, int component){
    return fieldRef_.get(index, component);
  }

  template<typename MeshIndex> KOKKOS_FUNCTION
  T& operator()(MeshIndex index, int component){
    return fieldRef_.operator()(index, component);
  }

private:
  bool is_read(){
    return std::is_same<SCOPE, READ>::value || std::is_same<SCOPE, READ_WRITE>::value;
  }

  bool is_write(){
    return std::is_same<SCOPE, WRITE>::value || std::is_same<SCOPE, READ_WRITE>::value;
  }

  stk::mesh::NgpField<T>& fieldRef_;
  const bool is_copy_constructed_{false};
};


template <typename SCOPE, typename T>
class SmartFieldRef<HOST, SCOPE, T>{
public:
  SmartFieldRef(stk::mesh::HostField<T>& ngpField):
    fieldRef_(ngpField){}

  SmartFieldRef(const SmartFieldRef& src):
    fieldRef_(src.fieldRef_),
    is_copy_constructed_(true)
  {
    if(is_read())
      fieldRef_.sync_to_device();
    else
      fieldRef_.clear_sync_state();
  }

  // device implementations should only ever execute inside a kokkos::paralle_for
  // and hence be captured by a lambda.
  // Therefore we only ever need to sync copies that will have been snatched up
  // through lambda capture.
  ~SmartFieldRef(){
    if(is_copy_constructed_ && is_write()){
      fieldRef_.modify_on_device();
    }
  }

  unsigned get_ordinal() const{
    return fieldRef_.get_ordinal();
  }

  T& get(stk::mesh::FastMeshIndex index, int component){
    return fieldRef_.get(index, component);
  }

  template<typename MeshIndex>
  T& get(MeshIndex index, int component){
    return fieldRef_.get(index, component);
  }

  T& operator()(stk::mesh::FastMeshIndex index, int component){
    return fieldRef_.get(index, component);
  }

  template<typename MeshIndex>
  T& operator()(MeshIndex index, int component){
    return fieldRef_.operator()(index, component);
  }

private:
  bool is_read(){
    return std::is_same<SCOPE, READ>::value || std::is_same<SCOPE, READ_WRITE>::value;
  }

  bool is_write(){
    return std::is_same<SCOPE, WRITE>::value || std::is_same<SCOPE, READ_WRITE>::value;
  }

  stk::mesh::HostField<T>& fieldRef_;
  const bool is_copy_constructed_{false};
};
}

#endif
