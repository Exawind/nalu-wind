// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <Kokkos_Macros.hpp>
#include <gtest/gtest.h>
#include "UnitTestUtils.h"
#include <stk_mesh/base/GetNgpField.hpp>
#include "ngp_utils/SmartFieldRef.h"

template <typename T>
class SmartFieldPtr{
public:
  SmartFieldPtr(stk::mesh::NgpField<T>& fieldRef):fieldPtr_(fieldRef){}

  SmartFieldPtr(const SmartFieldPtr& src): fieldPtr_(src.fieldPtr_){
    fieldPtr_.sync_to_device();
  }

  KOKKOS_FUNCTION
  unsigned get_ordinal() const{
    return fieldPtr_.get_ordinal();
  }

  ~SmartFieldPtr(){
      fieldPtr_.modify_on_device();
  }
  stk::mesh::NgpField<T>& fieldPtr_;
};


template<typename T>
void lambda_impl(T& ptr){
  Kokkos::parallel_for(1,
                       KOKKOS_LAMBDA(int){
                           ptr.get_ordinal();
                       });
}

namespace sierra::nalu{
TEST_F(Hex8Mesh, SmartFieldPtr){
  fill_mesh_and_initialize_test_fields();

  auto* field = fieldManager->get_field_ptr<ScalarFieldType>("scalarQ");

  stk::mesh::NgpField<double>& ngpField =
    stk::mesh::get_updated_ngp_field<double>(*field);

  ngpField.modify_on_host();

  ASSERT_TRUE(ngpField.need_sync_to_device());

  auto sPtr = SmartFieldPtr(ngpField);
  lambda_impl(sPtr);

  EXPECT_FALSE(ngpField.need_sync_to_device());
  EXPECT_TRUE(ngpField.need_sync_to_host());

}

TEST_F(Hex8Mesh, SmartFieldRef){
  fill_mesh_and_initialize_test_fields();

  auto* field = fieldManager->get_field_ptr<ScalarFieldType>("scalarQ");

  stk::mesh::NgpField<double>& ngpField =
    stk::mesh::get_updated_ngp_field<double>(*field);

  ngpField.modify_on_host();

  ASSERT_TRUE(ngpField.need_sync_to_device());

  auto sPtr = nalu_ngp::DeviceSmartFieldRef(ngpField, nalu_ngp::Scope::READWRITE);
  lambda_impl(sPtr);

  EXPECT_FALSE(ngpField.need_sync_to_device());
  EXPECT_TRUE(ngpField.need_sync_to_host());

}
}
