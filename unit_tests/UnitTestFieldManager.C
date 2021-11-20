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
#include "FieldManager.h"
#include <memory>
#include <stdexcept>

namespace sierra {
namespace nalu {
namespace {

class FieldManagerTest : public testing::Test
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

TEST_F(FieldManagerTest, nameIsEnoughInfoToRegisterAField)
{
  std::string name = "velocity";
  FieldManager fm(*meta_);
  EXPECT_FALSE(fm.field_exists(name));

  auto ptr = fm.register_field(name, meta_->get_parts());

  // check that field is on the mesh
  const auto findFieldPtr =
    meta_->get_field<VectorFieldType>(stk::topology::NODE_RANK, name);
  EXPECT_EQ(findFieldPtr, std::get<VectorFieldType*>(ptr));
  EXPECT_TRUE(fm.field_exists(name));
}

TEST_F(FieldManagerTest, throwsForFieldNotInDatabase)
{
  FieldManager f(*meta_);
  EXPECT_THROW(f.field_exists("acrazyqoi"), std::out_of_range);
}

TEST_F(FieldManagerTest, canRegisterDifferentFieldTypesThroughOneInterface)
{
  const std::string vectorName = "velocity";
  const std::string scalarName = "temperature";
  FieldManager f(*meta_);
  EXPECT_FALSE(f.field_exists(vectorName));
  EXPECT_FALSE(f.field_exists(scalarName));
  EXPECT_NO_THROW(f.register_field(vectorName, meta_->get_parts()));
  EXPECT_NO_THROW(f.register_field(scalarName, meta_->universal_part()));
  EXPECT_TRUE(f.field_exists(vectorName));
  EXPECT_TRUE(f.field_exists(scalarName));
}

TEST_F(FieldManagerTest, fieldCanBeRegisteredMultipleTimes)
{
  const std::string name = "velocity";
  FieldManager fm(*meta_);
  EXPECT_FALSE(fm.field_exists(name));
  EXPECT_NO_THROW(fm.register_field(name, meta_->get_parts()));
  EXPECT_NO_THROW(fm.register_field(name, meta_->universal_part()));
  EXPECT_TRUE(fm.field_exists(name));
}
} // namespace
} // namespace nalu
} // namespace sierra