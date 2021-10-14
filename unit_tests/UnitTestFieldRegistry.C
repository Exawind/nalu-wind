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

TEST_F(FieldRegistryTest, entityHasInfoToRegisterAField)
{
  const std::string name = "velocity";
  FieldRegistry fieldRegistry;
  auto* field = fieldRegistry.get_field_entity(name);
  EXPECT_EQ(stk::topology::NODE_RANK, field->rank);

  const auto parts = meta_->get_parts();
  const auto id = &(meta_->declare_field<VectorFieldType>(field->rank, name));

  for (auto&& part : parts)
    stk::mesh::put_field_on_mesh(*id, *part, nullptr);

  // check that field is on the mesh
  const auto findFieldPtr =
    meta_->get_field<VectorFieldType>(field->rank, name);
  EXPECT_TRUE(findFieldPtr);
}

// TEST_F(FieldRegistryTest, canQueryDifferentFieldTypesThroughOneInterface)
//{
//  const std::string vectorName = "velocity";
//  const std::string scalarName = "temperature";
//  FieldRegistry fieldRegistry;
//  const auto vectorEntity = fieldRegistry.get_field_entity(vectorName);
//  const auto scalarEntity = fieldRegistry.get_field_entity(scalarName);
//  EXPECT_EQ(stk::topology::NODE_RANK, vectorEntity.rank);
//  EXPECT_EQ(stk::topology::NODE_RANK, scalarEntity.rank);
//}

} // namespace
} // namespace nalu
} // namespace sierra