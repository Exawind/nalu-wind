// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gtest/gtest.h"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_io/StkMeshIoBroker.hpp"
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
    meta_ = std::make_unique<stk::mesh::MetaData>();
    bulk_ = std::make_unique<stk::mesh::BulkData>(*meta_, MPI_COMM_WORLD);
    stk::io::StkMeshIoBroker broker;
    broker.set_bulk_data(*bulk_);
    broker.add_mesh_database("generated:8x8x8", stk::io::READ_MESH);
    key_ = "velocity";
  }
  std::unique_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  std::string key_;
};

TEST_F(FieldRegistryTest, allDataNeededToAddFieldIsOnDefintion)
{
  auto def = FieldRegistry::query(key_);

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
  auto def = FieldRegistry::query(key_);
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