#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <stdexcept>
#include <tuple>

#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_util/parallel/Parallel.hpp>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>

#include <AlgTraits.h>
#include <EigenDecomposition.h>
#include <NaluEnv.h>

#include "UnitTestUtils.h"

#if !defined(KOKKOS_ENABLE_GPU)

namespace {

std::vector<double>
calculate_mij_tensor(
  sierra::nalu::MasterElement& me, std::vector<double>& ws_coords)
{
  int gradSize = me.num_integration_points() * me.nodesPerElement_ * me.nDim_;
  std::vector<double> ws_dndx(gradSize);
  std::vector<double> ws_deriv(gradSize);
  const sierra::nalu::SharedMemView<double**> coords(
    ws_coords.data(), me.nodesPerElement_, me.nDim_);
  sierra::nalu::SharedMemView<double***> dndx(
    ws_dndx.data(), me.num_integration_points(), me.nodesPerElement_, me.nDim_);
  sierra::nalu::SharedMemView<double***> deriv(
    ws_deriv.data(), me.num_integration_points(), me.nodesPerElement_,
    me.nDim_);
  me.grad_op(coords, dndx, deriv);

  int metricSize = me.nDim_ * me.nDim_ * me.num_integration_points();
  std::vector<double> mij_tensor(metricSize);
  me.Mij(ws_coords.data(), mij_tensor.data(), ws_deriv.data());

  return mij_tensor;
}

using VectorFieldType = stk::mesh::Field<double>;

void
test_metric_for_topo_2D(stk::topology topo, double tol)
{
  int dim = topo.dimension();
  ASSERT_EQ(dim, 2);

  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(dim);
  auto bulk = meshBuilder.create();
  auto& meta = bulk->mesh_meta_data();
  meta.use_simple_fields();

  stk::mesh::Entity elem =
    unit_test_utils::create_one_reference_element(*bulk, topo);

  auto* mescs =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);

  // apply some arbitrary linear map the reference element
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double Q[4] = {
    1.0 + std::abs(coeff(rng)), coeff(rng), coeff(rng),
    1.0 + std::abs(coeff(rng))};

  double Qt[4];
  sierra::nalu::transpose22(Q, Qt);

  double gij_exact[4];
  sierra::nalu::mxm22(Q, Qt, gij_exact);

  // Compute eigendecomposition
  double mij_ev[2][2];
  double mij_evals[2][2];
  double(&gij_exact_pt)[2][2] = reinterpret_cast<double(&)[2][2]>(gij_exact);
  sierra::nalu::EigenDecomposition::sym_diagonalize(
    gij_exact_pt, mij_ev, mij_evals);

  // Construct Mij
  double mij_exact[2][2];
  for (unsigned i = 0; i < 2; i++)
    for (unsigned j = 0; j < 2; j++)
      mij_exact[i][j] =
        mij_ev[i][0] * mij_ev[j][0] * stk::math::sqrt(mij_evals[0][0]) +
        mij_ev[i][1] * mij_ev[j][1] * stk::math::sqrt(mij_evals[1][1]);

  const auto& coordField =
    *static_cast<const VectorFieldType*>(meta.coordinate_field());
  std::vector<double> ws_coords(topo.num_nodes() * dim);
  const auto* nodes = bulk->begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, nodes[j]);
    sierra::nalu::matvec22(Q, coords, &ws_coords[j * dim]);
  }

  std::vector<double> mij_tensor = calculate_mij_tensor(*mescs, ws_coords);

  for (int ip = 0; ip < mescs->num_integration_points(); ++ip) {
    for (unsigned i = 0; i < 2; i++) {
      for (unsigned j = 0; j < 2; j++) {
        EXPECT_NEAR(mij_tensor[4 * ip + (i * 2 + j)], mij_exact[i][j], tol);
      }
    }
  }
}

void
test_metric_for_topo_3D(stk::topology topo, double tol)
{
  int dim = topo.dimension();
  ASSERT_EQ(dim, 3);

  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(dim);
  auto bulk = meshBuilder.create();
  auto& meta = bulk->mesh_meta_data();
  meta.use_simple_fields();

  stk::mesh::Entity elem =
    unit_test_utils::create_one_reference_element(*bulk, topo);

  auto* mescs =
    sierra::nalu::MasterElementRepo::get_surface_master_element_on_host(topo);

  // apply some arbitrary linear map the reference element
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double Q[9] = {1.0 + std::abs(coeff(rng)), coeff(rng), coeff(rng), coeff(rng),
                 1.0 + std::abs(coeff(rng)), coeff(rng), coeff(rng), coeff(rng),
                 1.0 + std::abs(coeff(rng))};

  double Qt[9];
  sierra::nalu::transpose33(Q, Qt);

  double gij_exact[9];
  sierra::nalu::mxm33(Q, Qt, gij_exact);

  // Compute eigendecomposition
  double mij_ev[3][3];
  double mij_evals[3][3];
  double(&gij_exact_pt)[3][3] = reinterpret_cast<double(&)[3][3]>(gij_exact);
  sierra::nalu::EigenDecomposition::sym_diagonalize(
    gij_exact_pt, mij_ev, mij_evals);

  // Construct Mij
  double mij_exact[3][3];
  for (unsigned i = 0; i < 3; i++)
    for (unsigned j = 0; j < 3; j++)
      mij_exact[i][j] =
        mij_ev[i][0] * mij_ev[j][0] * stk::math::sqrt(mij_evals[0][0]) +
        mij_ev[i][1] * mij_ev[j][1] * stk::math::sqrt(mij_evals[1][1]) +
        mij_ev[i][2] * mij_ev[j][2] * stk::math::sqrt(mij_evals[2][2]);

  const auto& coordField =
    *static_cast<const VectorFieldType*>(meta.coordinate_field());
  std::vector<double> ws_coords(topo.num_nodes() * dim);
  const auto* nodes = bulk->begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, nodes[j]);
    sierra::nalu::matvec33(Q, coords, &ws_coords[j * dim]);
  }

  std::vector<double> mij_tensor = calculate_mij_tensor(*mescs, ws_coords);

  for (int ip = 0; ip < mescs->num_integration_points(); ++ip) {
    for (unsigned i = 0; i < 3; i++) {
      for (unsigned j = 0; j < 3; j++) {
        EXPECT_NEAR(mij_tensor[9 * ip + (i * 3 + j)], mij_exact[i][j], tol);
      }
    }
  }
}

} // namespace

TEST(MijTensor, tri3)
{
  test_metric_for_topo_2D(stk::topology::TRIANGLE_3_2D, 1.0e-10);
}

TEST(MijTensor, quad4)
{
  test_metric_for_topo_2D(stk::topology::QUADRILATERAL_4_2D, 1.0e-10);
}

TEST(MijTensor, tet4)
{
  test_metric_for_topo_3D(stk::topology::TET_4, 1.0e-10);
}

TEST(MijTensor, wedge6)
{
  test_metric_for_topo_3D(stk::topology::WEDGE_6, 1.0e-10);
}

TEST(MijTensor, hex8)
{
  test_metric_for_topo_3D(stk::topology::HEX_8, 1.0e-10);
}
#endif
