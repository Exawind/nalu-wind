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
class DeviceSmartFieldRef{
public:
  DeviceSmartFieldRef(stk::mesh::NgpField<T>& ngpField, Scope scope):fieldRef_(ngpField),scope_(scope){}

  DeviceSmartFieldRef(const DeviceSmartFieldRef& src):fieldRef_(src.fieldRef_), scope_(src.scope_), is_copy_constructed_(true)
  {
    if(scope_ == Scope::WRITE)
      fieldRef_.clear_sync_state();
    else
      fieldRef_.sync_to_device();
  }

  ~DeviceSmartFieldRef(){
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

  stk::mesh::NgpField<T>& fieldRef_;
  const Scope scope_;
  const bool is_copy_constructed_{false};
};


}

#endif
