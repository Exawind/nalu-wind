#include "gtest/gtest.h"
#include "SideWriter.h"

#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/Field.hpp"
#include <yaml-cpp/yaml.h>

namespace sierra {
namespace nalu {
class SideWriterFixture : public ::testing::Test
{
public:
  SideWriterFixture()
  {
    auto spatialDimension = 3u;
    stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
    meshBuilder.set_spatial_dimension(spatialDimension);
    meshBuilder.set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA);
    bulk = meshBuilder.create();
    meta = &bulk->mesh_meta_data();
    stk::io::StkMeshIoBroker io(bulk->parallel());

    test_field = &meta->declare_field<stk::mesh::Field<double>>(
      stk::topology::NODE_RANK, "test");
    test_vector_field =
      &meta->declare_field<stk::mesh::Field<double, stk::mesh::Cartesian3d>>(
        stk::topology::NODE_RANK, "test_vector");

    double minus_one = -1;
    stk::mesh::put_field_on_mesh(
      *test_field, meta->universal_part(), 1, &minus_one);

    stk::mesh::put_field_on_mesh(
      *test_vector_field, meta->universal_part(), 3, &minus_one);

    const std::string name = "generated:3x3x3|sideset:xXyYzZ";
    io.set_bulk_data(*bulk);
    io.add_mesh_database(name, stk::io::READ_MESH);
    io.create_input_mesh();
    io.populate_bulk_data();
    stk::io::put_io_part_attribute(meta->universal_part());
  }

  stk::mesh::MetaData* meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  // stk::io::StkMeshIoBroker io;
  stk::mesh::Field<double>* test_field;
  stk::mesh::Field<double, stk::mesh::Cartesian3d>* test_vector_field;
};

TEST_F(SideWriterFixture, side)
{

  std::vector<const stk::mesh::Part*> sides{
    meta->get_part("surface_1"), meta->get_part("surface_2")};
  SideWriter side_io(
    *bulk, sides, {test_field, test_vector_field}, "test_output/file.e");

  auto& coord_field =
    *static_cast<const stk::mesh::Field<double, stk::mesh::Cartesian3d>*>(
      meta->coordinate_field());

  const auto& all_node_buckets =
    bulk->get_buckets(stk::topology::NODE_RANK, meta->universal_part());
  for (const auto* ib : all_node_buckets) {
    for (const auto node : *ib) {
      const double x = stk::mesh::field_data(coord_field, node)[0];
      const double y = stk::mesh::field_data(coord_field, node)[1];
      const double z = stk::mesh::field_data(coord_field, node)[2];

      *stk::mesh::field_data(*test_field, node) = x;

      stk::mesh::field_data(*test_vector_field, node)[0] = -y;
      stk::mesh::field_data(*test_vector_field, node)[1] = x;
      stk::mesh::field_data(*test_vector_field, node)[2] = z;
    }
  }
  side_io.write_database_data(0.);

  for (const auto* ib : all_node_buckets) {
    for (const auto node : *ib) {
      const double x = stk::mesh::field_data(coord_field, node)[0];
      const double y = stk::mesh::field_data(coord_field, node)[1];
      const double z = stk::mesh::field_data(coord_field, node)[2];

      *stk::mesh::field_data(*test_field, node) = y;

      stk::mesh::field_data(*test_vector_field, node)[0] = -z;
      stk::mesh::field_data(*test_vector_field, node)[1] = y;
      stk::mesh::field_data(*test_vector_field, node)[2] = x;
    }
  }
  side_io.write_database_data(1.);
}

TEST(SideWriterContainerTest, load)
{
  const char* input = R"test(sideset_writers:
    - name: w1
      output_data_base_name: w1.exo
      output_frequency: 1
      target_name: side_1
      output_variables: [velocity]
    - name: w2
      output_data_base_name: w2.exo
      output_frequency: 2
      target_name: [side_2]
      output_variables: [pressure])test";

  const YAML::Node y_node = YAML::Load(input);
  SideWriterContainer container;
  ASSERT_NO_THROW(container.load(y_node));
  EXPECT_EQ(container.number_of_writers(), 2);
}

} // namespace nalu
} // namespace sierra