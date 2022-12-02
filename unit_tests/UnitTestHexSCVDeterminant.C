#include <gtest/gtest.h>
#include <limits>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <master_element/MasterElement.h>
#include <master_element/Hex8GeometryFunctions.h>

#include <random>

#include "UnitTestUtils.h"

namespace {

double
determinant33(const double* mat)
{
  enum {
    XX = 0,
    XY = 1,
    XZ = 2,
    YX = 3,
    YY = 4,
    YZ = 5,
    ZX = 6,
    ZY = 7,
    ZZ = 8
  };
  return (
    mat[XX] * (mat[YY] * mat[ZZ] - mat[YZ] * mat[ZY]) -
    mat[XY] * (mat[YX] * mat[ZZ] - mat[YZ] * mat[ZX]) +
    mat[XZ] * (mat[YX] * mat[ZY] - mat[YY] * mat[ZX]));
}

void
check_HexSCV_determinant(const stk::mesh::BulkData& bulk)
{
  typedef stk::mesh::Field<double, stk::mesh::Cartesian> CoordFieldType;
  CoordFieldType* coordField = bulk.mesh_meta_data().get_field<CoordFieldType>(
    stk::topology::NODE_RANK, "coordinates");
  EXPECT_TRUE(coordField != nullptr);

  stk::mesh::EntityVector elems;
  stk::mesh::get_entities(bulk, stk::topology::ELEM_RANK, elems);

  const double tolerance = std::numeric_limits<double>::epsilon();

  stk::topology hex8 = stk::topology::HEX_8;

  const unsigned spatialDim = bulk.mesh_meta_data().spatial_dimension();
  std::vector<double> hex8_node_coords(hex8.num_nodes() * spatialDim, 0.0);
  std::vector<double> hex8_scvolumes(hex8.num_nodes(), 0.0);
  sierra::nalu::SharedMemView<double**> node_coords(
    hex8_node_coords.data(), hex8.num_nodes(), spatialDim);
  sierra::nalu::SharedMemView<double*> scvolumes(
    hex8_scvolumes.data(), hex8.num_nodes());

  sierra::nalu::HexSCV hexSCV;
  double error[1] = {0};
  for (stk::mesh::Entity elem : elems) {
    EXPECT_EQ(hex8, bulk.bucket(elem).topology());

    const stk::mesh::Entity* nodes = bulk.begin_nodes(elem);
    unsigned numNodes = bulk.num_nodes(elem);
    for (unsigned i = 0; i < numNodes; ++i) {
      double* nodeCoords = stk::mesh::field_data(*coordField, nodes[i]);

      for (unsigned d = 0; d < spatialDim; ++d) {
        node_coords(i, d) = nodeCoords[d];
      }
    }

    hexSCV.determinant(node_coords, scvolumes);

    EXPECT_EQ(0, error[0]);

    for (unsigned i = 0; i < hex8.num_nodes(); ++i) {
      EXPECT_NEAR(0.125, scvolumes(i), tolerance);
    }
  }
}

} // namespace

TEST(HexSCV, determinant)
{
  stk::ParallelMachine comm = MPI_COMM_WORLD;

  unsigned spatialDimension = 3;
  stk::mesh::MeshBuilder meshBuilder(comm);
  meshBuilder.set_spatial_dimension(spatialDimension);
  auto bulk = meshBuilder.create();

  unit_test_utils::fill_mesh_1_elem_per_proc_hex8(*bulk);

  check_HexSCV_determinant(*bulk);
}

TEST(HexSCV, grandyvol)
{
  stk::ParallelMachine comm = MPI_COMM_WORLD;

  unsigned spatialDimension = 3;
  stk::mesh::MeshBuilder meshBuilder(comm);
  meshBuilder.set_spatial_dimension(spatialDimension);
  auto bulk = meshBuilder.create();

  unit_test_utils::fill_mesh_1_elem_per_proc_hex8(*bulk);
  const auto& coordField = *static_cast<const VectorFieldType*>(
    bulk->mesh_meta_data().coordinate_field());

  double v_coords[8][3];

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double Q[9] = {2.0 + coeff(rng), coeff(rng),       coeff(rng),
                 coeff(rng),       2.5 + coeff(rng), coeff(rng),
                 coeff(rng),       coeff(rng),       3.0 + coeff(rng)};
  double detQ = determinant33(Q);

  for (auto* ib : bulk->get_buckets(
         stk::topology::ELEM_RANK, bulk->mesh_meta_data().universal_part())) {
    const auto& b = *ib;
    for (size_t k = 0u; k < b.size(); ++k) {
      const auto* node_rels = bulk->begin_nodes(b[k]);
      for (int n = 0; n < 8; ++n) {
        for (int j = 0; j < 3; ++j) {
          v_coords[n][j] = 0.0;
          for (int i = 0; i < 3; ++i) {
            v_coords[n][j] +=
              Q[3 * j + i] * stk::mesh::field_data(coordField, node_rels[n])[i];
          }
        }
      }

      double exactVol = detQ;
      // start caliper
      double volGrandy = sierra::nalu::hex_volume_grandy(v_coords);
      // end caliper
      EXPECT_NEAR(volGrandy, exactVol, tol);
    }
  }
}
