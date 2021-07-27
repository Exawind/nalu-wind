#include "gtest/gtest.h"
#include "SideWriter.h"

#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {
class SideWriterFixture : public ::testing::Test
{
public:
  SideWriterFixture()
    : meta(3u),
      bulk(meta, MPI_COMM_WORLD, stk::mesh::BulkData::NO_AUTO_AURA),
      io(bulk.parallel()),
      test_field(meta.declare_field<stk::mesh::Field<double>>(
        stk::topology::NODE_RANK, "test"))

  {
    double minus_one = -1;
    stk::mesh::put_field_on_mesh(
      test_field, meta.universal_part(), 1, &minus_one);
  }
  stk::mesh::MetaData meta;
  stk::mesh::BulkData bulk;
  stk::io::StkMeshIoBroker io;
  stk::mesh::Field<double>& test_field;
};

TEST_F(SideWriterFixture, side)
{
  const std::string name = "generated:3x3x3|sideset:xXyYzZ";
  io.set_bulk_data(bulk);
  io.add_mesh_database(name, stk::io::READ_MESH);
  io.create_input_mesh();
  io.populate_bulk_data();
  stk::io::put_io_part_attribute(meta.universal_part());
  std::vector<const stk::mesh::Part*> sides{
    meta.get_part("surface_1"), meta.get_part("surface_2")};
  auto& coord_field =
    *static_cast<const stk::mesh::Field<double, stk::mesh::Cartesian3d>*>(
      meta.coordinate_field());
  const auto& all_node_buckets =
    bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part());
  for (const auto* ib : all_node_buckets) {
    for (const auto node : *ib) {
      *stk::mesh::field_data(test_field, node) =
        stk::mesh::field_data(coord_field, node)[0];
    }
  }
  SideWriter side_io(bulk, sides, {&test_field}, "file.e");
  side_io.write_database_data(0.);

  for (const auto* ib : all_node_buckets) {
    for (const auto node : *ib) {
      *stk::mesh::field_data(test_field, node) = stk::mesh::field_data(coord_field, node)[1];
    }
  }
  side_io.write_database_data(1.);
}

} // namespace nalu
} // namespace sierra