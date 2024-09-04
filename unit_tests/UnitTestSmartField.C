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
#include "SmartField.h"

class TestSmartField : public Hex8Mesh
{
public:
protected:
  void SetUp()
  {
    fill_mesh_and_initialize_test_fields();

    auto* field = fieldManager->get_field_ptr<double>("scalarQ");

    ngpField_ = &stk::mesh::get_updated_ngp_field<double>(*field);

    initSyncsHost_ = ngpField_->num_syncs_to_host();
    initSyncsDevice_ = ngpField_->num_syncs_to_device();
  }

  stk::mesh::NgpField<double>* ngpField_;
  int initSyncsHost_{0};
  int initSyncsDevice_{0};
};
//*****************************************************************************
// Free functions for execution on device
//*****************************************************************************
template <typename T>
void
lambda_ordinal(T& ptr)
{
  Kokkos::parallel_for(1, KOKKOS_LAMBDA(int) { ptr.get_ordinal(); });
}

template <typename T>
void
lambda_loop_assign(
  stk::mesh::BulkData& bulk,
  stk::mesh::PartVector partVec,
  T& ptr,
  double val = 300.0)
{
  stk::mesh::NgpMesh& ngpMesh = stk::mesh::get_updated_ngp_mesh(bulk);
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec);
  stk::mesh::for_each_entity_run(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity) {
      ptr(entity, 0) = val;
    });
}

template <typename T>
void
lambda_loop_performance(
  stk::mesh::BulkData& bulk,
  stk::mesh::PartVector partVec,
  T& ptr,
  double val = 300.0)
{
  stk::mesh::NgpMesh& ngpMesh = stk::mesh::get_updated_ngp_mesh(bulk);
  stk::mesh::Selector sel = stk::mesh::selectUnion(partVec);
  stk::mesh::for_each_entity_run(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity) {
      for (int i = 0; i < 1e6; ++i)
        ptr(entity, 0) = val;
    });
}

//*****************************************************************************
// Tests
//*****************************************************************************
namespace sierra::nalu {
using namespace tags;

TEST_F(TestSmartField, device_read_write_mod_sync_with_lambda)
{
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = MakeSmartField<DEVICE, READ_WRITE>()(*ngpField_);
  lambda_ordinal(sPtr);

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_TRUE(ngpField_->need_sync_to_host());
  EXPECT_EQ(initSyncsDevice_ + 1, ngpField_->num_syncs_to_device());
  EXPECT_EQ(initSyncsHost_ + 0, ngpField_->num_syncs_to_host());
}

TEST_F(TestSmartField, device_write_clear_mod_with_lambda)
{
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = MakeSmartField<DEVICE, WRITE_ALL>()(*ngpField_);
  lambda_ordinal(sPtr);

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_TRUE(ngpField_->need_sync_to_host());
  EXPECT_EQ(initSyncsDevice_ + 0, ngpField_->num_syncs_to_device());
  EXPECT_EQ(initSyncsHost_ + 0, ngpField_->num_syncs_to_host());
}

TEST_F(TestSmartField, device_read_mod_no_sync_with_lambda)
{
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = MakeSmartField<DEVICE, READ>()(*ngpField_);
  lambda_ordinal(sPtr);

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_FALSE(ngpField_->need_sync_to_host());
  EXPECT_EQ(initSyncsDevice_ + 1, ngpField_->num_syncs_to_device());
  EXPECT_EQ(initSyncsHost_ + 0, ngpField_->num_syncs_to_host());
}

TEST_F(TestSmartField, device_performance_smart_field)
{
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = MakeSmartField<DEVICE, READ_WRITE>()(*ngpField_);

  double assignmentValue = 300.0;
  lambda_loop_performance(*bulk, partVec, sPtr, assignmentValue);
}

TEST_F(TestSmartField, device_performance_ngp_field)
{
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());
  ngpField_->sync_to_device();

  double assignmentValue = 300.0;
  lambda_loop_performance(*bulk, partVec, *ngpField_, assignmentValue);
}

TEST_F(TestSmartField, update_field_on_device_check_on_host)
{
  ngpField_->modify_on_host();

  ASSERT_TRUE(ngpField_->need_sync_to_device());

  auto sPtr = MakeSmartField<DEVICE, READ_WRITE>()(*ngpField_);

  double assignmentValue = 300.0;
  lambda_loop_assign(*bulk, partVec, sPtr, assignmentValue);

  // Check field values on host using standard bucket loop
  // Do it inside brackets so fieldRef will destruct
  {
    double sum = 0.0;
    int counter = 0;
    auto* field = fieldManager->get_field_ptr<double>("scalarQ");
    auto fieldRef =
      sierra::nalu::MakeSmartField<tags::LEGACY, tags::READ>()(field);
    stk::mesh::Selector sel = stk::mesh::selectUnion(partVec);
    const auto& buckets = bulk->get_buckets(stk::topology::NODE_RANK, sel);
    for (auto b : buckets) {
      for (size_t in = 0; in < b->size(); in++) {
        auto node = (*b)[in];
        sum += *fieldRef.get(node);
        counter++;
      }
    }
    EXPECT_NEAR(assignmentValue * counter, sum, 1e-12);
  }

  EXPECT_FALSE(ngpField_->need_sync_to_device());
  EXPECT_FALSE(ngpField_->need_sync_to_host());
  EXPECT_EQ(initSyncsDevice_ + 1, ngpField_->num_syncs_to_device());
  EXPECT_EQ(initSyncsHost_ + 1, ngpField_->num_syncs_to_host());
}

} // namespace sierra::nalu
