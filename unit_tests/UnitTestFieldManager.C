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
    stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
    builder.set_spatial_dimension(3);
    meta_ = builder.create_meta_data();
    key_ = "velocity";
  }
  stk::mesh::MetaData& meta() { return *(meta_.get()); }
  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::string key_;
};

TEST_F(FieldManagerTest, nameIsEnoughInfoToRegisterAField)
{
  const int num_states = 2;
  std::string name = "velocity";
  FieldManager fm(meta(), num_states);
  EXPECT_FALSE(fm.field_exists(name));

  auto ptr = fm.register_field(name, meta().get_parts());

  // check that field is on the mesh
  const auto findFieldPtr =
    meta().get_field<VectorFieldType>(stk::topology::NODE_RANK, name);
  EXPECT_EQ(findFieldPtr, std::get<VectorFieldType*>(ptr));
  EXPECT_TRUE(fm.field_exists(name));

  auto ptr2 = fm.get_field_ptr<VectorFieldType*>(name);
  EXPECT_EQ(findFieldPtr, ptr2);
}

TEST_F(FieldManagerTest, throwsForFieldNotInDatabase)
{
  const int num_states = 2;
  FieldManager f(meta(), num_states);
  EXPECT_THROW(f.field_exists("acrazyqoi"), std::runtime_error);
}

TEST_F(FieldManagerTest, canRegisterDifferentFieldTypesThroughOneInterface)
{
  const std::string vectorName = "velocity";
  const std::string scalarName = "temperature";
  const int num_states = 2;
  FieldManager f(meta(), num_states);
  EXPECT_FALSE(f.field_exists(vectorName));
  EXPECT_FALSE(f.field_exists(scalarName));
  EXPECT_NO_THROW(f.register_field(vectorName, meta().get_parts()));
  EXPECT_NO_THROW(f.register_field(scalarName, meta().universal_part()));
  EXPECT_TRUE(f.field_exists(vectorName));
  EXPECT_TRUE(f.field_exists(scalarName));
}

TEST_F(FieldManagerTest, fieldCanBeRegisteredMultipleTimes)
{
  const std::string name = "velocity";
  const int num_states = 3;
  FieldManager fm(meta(), num_states);
  EXPECT_FALSE(fm.field_exists(name));
  EXPECT_NO_THROW(fm.register_field(name, meta().get_parts()));
  EXPECT_NO_THROW(fm.register_field(name, meta().universal_part()));
  EXPECT_TRUE(fm.field_exists(name));
}

TEST_F(FieldManagerTest, undefinedFieldCantBeRegistered)
{
  const std::string name = "fields_of_gold";
  const int num_states = 3;
  FieldManager fm(meta(), num_states);
  EXPECT_THROW(
    fm.register_field(name, meta().universal_part()), std::runtime_error);
}

TEST_F(FieldManagerTest, fieldStateCanBeSelected)
{
  const std::string name = "velocity";
  const int numStates = 3;
  FieldManager fm(meta(), numStates);
  fm.register_field(name, meta().universal_part());
  // clang-format off
  const auto np1 = fm.get_field_ptr<VectorFieldType*>(name, stk::mesh::StateNP1);
  const auto n =   fm.get_field_ptr<VectorFieldType*>(name, stk::mesh::StateN);
  const auto nm1 = fm.get_field_ptr<VectorFieldType*>(name, stk::mesh::StateNM1);
  // clang-format on
  EXPECT_TRUE(np1 != nullptr);
  EXPECT_TRUE(n!= nullptr);
  EXPECT_TRUE(nm1 != nullptr);
  EXPECT_TRUE(np1 != n);
  EXPECT_TRUE(np1 != nm1);
}
} // namespace
} // namespace nalu
} // namespace sierra
