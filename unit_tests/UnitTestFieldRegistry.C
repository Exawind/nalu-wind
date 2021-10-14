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
#include <FieldRegistry.h>
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
  }
  std::unique_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
};

TEST_F(FieldRegistryTest, nameIsEnoughInfoToRegisterAField)
{
  const std::string name = "velocity";
  FieldRegistry fieldRegistry;
  EXPECT_FALSE(fieldRegistry.field_exists(*meta_, name));

  fieldRegistry.register_field(*meta_, name, meta_->get_parts());

  // check that field is on the mesh
  const auto findFieldPtr =
    meta_->get_field<VectorFieldType>(stk::topology::NODE_RANK, name);
  EXPECT_TRUE(findFieldPtr);
  EXPECT_TRUE(fieldRegistry.field_exists(*meta_, name));
}

TEST_F(FieldRegistryTest, throwsForFieldNotInDatabase)
{
  FieldRegistry f;
  EXPECT_THROW(f.field_exists(*meta_, "acrazyqoi"), std::out_of_range);
}

TEST_F(FieldRegistryTest, canRegisterDifferentFieldTypesThroughOneInterface)
{
  const std::string vectorName = "velocity";
  const std::string scalarName = "temperature";
  FieldRegistry f;
  EXPECT_FALSE(f.field_exists(*meta_, vectorName));
  EXPECT_FALSE(f.field_exists(*meta_, scalarName));
  EXPECT_NO_THROW(f.register_field(*meta_, vectorName, meta_->get_parts()));
  EXPECT_NO_THROW(f.register_field(*meta_, scalarName, meta_->get_parts()));
  EXPECT_TRUE(f.field_exists(*meta_, vectorName));
  EXPECT_TRUE(f.field_exists(*meta_, scalarName));
}

TEST_F(FieldRegistryTest, fieldCanBeRegisteredMultipleTimes)
{
  const std::string name = "velocity";
  FieldRegistry fieldRegistry;
  EXPECT_FALSE(fieldRegistry.field_exists(*meta_, name));
  EXPECT_NO_THROW(
    fieldRegistry.register_field(*meta_, name, meta_->get_parts()));
  EXPECT_NO_THROW(
    fieldRegistry.register_field(*meta_, name, meta_->get_parts()));
  EXPECT_TRUE(fieldRegistry.field_exists(*meta_, name));
}

} // namespace
} // namespace nalu
} // namespace sierra