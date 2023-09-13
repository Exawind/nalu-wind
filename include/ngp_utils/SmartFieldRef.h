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
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpField.hpp>

namespace sierra::nalu::nalu_ngp{
struct READ{};

enum class Scope{
  READ,
  WRITE,
  READWRITE
};

template <typename T>
using DeviceField = stk::mesh::NgpField<T>;

template <typename T>
using HostField = stk::mesh::HostField<T>;

template <typename T, template<typename> typename FieldType>
class SmartFieldRef{
public:
  SmartFieldRef(FieldType<T>& ngpField, Scope scope):
    fieldRef_(ngpField),
    scope_(scope){}

  SmartFieldRef(const SmartFieldRef& src):
    fieldRef_(src.fieldRef_),
    scope_(src.scope_),
    is_copy_constructed_(true)
  {
    if(scope_ == Scope::WRITE)
      fieldRef_.clear_sync_state();
    else
      fieldRef_.sync_to_device();
  }

  ~SmartFieldRef(){
    if(is_copy_constructed_ && !scope_is(Scope::READ)){
      fieldRef_.modify_on_device();
    }
  }

  KOKKOS_FUNCTION
  unsigned get_ordinal() const{
    return fieldRef_.get_ordinal();
  }

private:
  bool scope_is(Scope test){
    return scope_ == test;
  }

  FieldType<T>& fieldRef_;
  const Scope scope_;
  const bool is_copy_constructed_{false};
};

template <typename T>
using DeviceSmartFieldRef = SmartFieldRef<T, DeviceField>;

}

#endif
