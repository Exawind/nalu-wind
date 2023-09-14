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

class TestSmartFieldRef : public Hex8Mesh{
protected:
  void SetUp(){
  fill_mesh_and_initialize_test_fields();

  auto* field = fieldManager->get_field_ptr<ScalarFieldType>("scalarQ");

  ngpField_ =
    &stk::mesh::get_updated_ngp_field<double>(*field);

  }
  stk::mesh::NgpField<double>* ngpField_;
};

template<typename T>
void lambda_impl(T& ptr){
  Kokkos::parallel_for(1,
                       KOKKOS_LAMBDA(int){
                           ptr.get_ordinal();
                       });
}

namespace sierra::nalu{
TEST_F(TestSmartFieldRef, device_read_write_mod_sync_with_lambda){
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  //TODO can we get rid of the double template param some how?
  auto sPtr = SmartFieldRef<DEVICE, READ_WRITE, double>(*ngpField_);
  lambda_impl(sPtr);

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_TRUE(ngpField_->need_sync_to_host());
}

TEST_F(TestSmartFieldRef, device_write_clear_mod_with_lambda){
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = SmartFieldRef<DEVICE, WRITE, double>(*ngpField_);
  lambda_impl(sPtr);

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_TRUE(ngpField_->need_sync_to_host());
}

TEST_F(TestSmartFieldRef, device_read_mod_no_sync_with_lambda){
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = SmartFieldRef<DEVICE, READ, double>(*ngpField_);
  lambda_impl(sPtr);

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_FALSE(ngpField_->need_sync_to_host());
}

}
