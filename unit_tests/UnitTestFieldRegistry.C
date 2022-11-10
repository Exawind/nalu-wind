// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "FieldRegistry.h"
#include <memory>
#include <stdexcept>

namespace sierra {
namespace nalu {
namespace {

class FieldRegistryTest : public testing::Test
{
protected:
  void SetUp()
  {
    stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
    meta_ = builder.create_meta_data();
    key_ = "velocity";
  }
  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::string key_;
};

TEST_F(FieldRegistryTest, allDataNeededToDeclareFieldIsKnownThroughQuery)
{
  const int num_states = 2;
  auto def = FieldRegistry::query(num_states, key_);

  ASSERT_NO_THROW(std::visit(
    [&](auto arg) {
      meta_->declare_field<typename decltype(arg)::FieldType>(
        arg.rank, key_, arg.num_states);
    },
    def));

  const auto findFieldPtr =
    meta_->get_field<VectorFieldType>(stk::topology::NODE_RANK, key_);
  // pointer to field is valid through stk api
  EXPECT_TRUE(findFieldPtr);
}

TEST_F(FieldRegistryTest, registeredFieldPointerCanBeStored)
{
  const int num_states = 3;
  auto def = FieldRegistry::query(num_states, key_);
  std::vector<FieldPointerTypes> field_pointers;

  EXPECT_EQ(0, field_pointers.size());
  ASSERT_NO_THROW(std::visit(
    [&](auto arg) {
      auto* ptr = &(meta_->declare_field<typename decltype(arg)::FieldType>(
        arg.rank, key_, arg.num_states));
      field_pointers.push_back(ptr);
    },
    def));

  // pointer storage has increased
  EXPECT_EQ(1, field_pointers.size());

  const auto findFieldPtr =
    meta_->get_field<VectorFieldType>(stk::topology::NODE_RANK, key_);
  // stk api and stored pointer are the same
  EXPECT_EQ(findFieldPtr, std::get<VectorFieldType*>(field_pointers[0]));
}

} // namespace
} // namespace nalu
} // namespace sierra
