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
  }
  std::unique_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
};

TEST_F(FieldRegistryTest, allDataNeededToAddFieldIsOnDefintion)
{

  std::string key = "velocity";
  auto def = FieldRegistry::query(key);

  ASSERT_NO_THROW(std::visit(
    [&](auto arg) {
      meta_->declare_field<typename decltype(arg)::FieldType>(
        arg.rank, key, arg.num_states);
    },
    def));
}

} // namespace
} // namespace nalu
} // namespace sierra