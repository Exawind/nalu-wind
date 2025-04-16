#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <random>
#include <tuple>
#include <ostream>
#include <memory>

#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/TensorOps.h>

#include <NaluEnv.h>
#include <AlgTraits.h>

#include "UnitTestUtils.h"

namespace {

std::pair<std::vector<DoubleType>, std::vector<DoubleType>>
calculate_metric_tensor(
  sierra::nalu::MasterElement& me, std::vector<DoubleType>& ws_coords)
{
  int gradSize = me.num_integration_points() * me.nodesPerElement_ * me.nDim_;
  std::vector<DoubleType> ws_dndx(gradSize);
  std::vector<DoubleType> ws_deriv(gradSize);
  const sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem>
    elemCoords(ws_coords.data(), me.nodesPerElement_, me.nDim_);
  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> dndx(
    ws_dndx.data(), me.num_integration_points(), me.nodesPerElement_, me.nDim_);
  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem> deriv(
    ws_deriv.data(), me.num_integration_points(), me.nodesPerElement_,
    me.nDim_);
  me.grad_op(elemCoords, dndx, deriv);

  int metricSize = me.nDim_ * me.nDim_ * me.num_integration_points();
  std::vector<DoubleType> ws_contravariant_metric_tensor(metricSize);
  std::vector<DoubleType> ws_covariant_metric_tensor(metricSize);
  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem>
    contravariant_metric_tensor(
      ws_contravariant_metric_tensor.data(), me.num_integration_points(),
      me.nDim_, me.nDim_);
  sierra::nalu::SharedMemView<DoubleType***, sierra::nalu::DeviceShmem>
    covariant_metric_tensor(
      ws_covariant_metric_tensor.data(), me.num_integration_points(), me.nDim_,
      me.nDim_);
  me.gij(
    elemCoords, contravariant_metric_tensor, covariant_metric_tensor, deriv);

  return {ws_contravariant_metric_tensor, ws_covariant_metric_tensor};
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
  bulk->mesh_meta_data().use_simple_fields();
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

  double metric_exact[4];
  sierra::nalu::mxm22(Q, Qt, metric_exact);

  const auto& coordField = *static_cast<const VectorFieldType*>(
    bulk->mesh_meta_data().coordinate_field());
  std::vector<DoubleType> ws_coords(topo.num_nodes() * dim);
  const sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem>
    coords(ws_coords.data(), topo.num_nodes(), dim);
  const auto* nodes = bulk->begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coord = stk::mesh::field_data(coordField, nodes[j]);
    double tmp[2];
    sierra::nalu::matvec22(Q, coord, tmp);
    coords(j, 0) = tmp[0];
    coords(j, 1) = tmp[1];
  }

  std::vector<DoubleType> contravariant_metric;
  std::vector<DoubleType> covariant_metric;
  std::tie(contravariant_metric, covariant_metric) =
    calculate_metric_tensor(*mescs, ws_coords);

  for (int ip = 0; ip < mescs->num_integration_points(); ++ip) {
    double identity[4] = {1.0, 0.0, 0.0, 1.0};
    DoubleType shouldBeIdentity[4];
    sierra::nalu::mxm22(
      &contravariant_metric[4 * ip], &covariant_metric[4 * ip],
      shouldBeIdentity);
    for (unsigned k = 0; k < 4; ++k) {
      EXPECT_NEAR(
        stk::simd::get_data(contravariant_metric[4 * ip + k], 0),
        metric_exact[k], tol);
      EXPECT_NEAR(
        stk::simd::get_data(shouldBeIdentity[k], 0), identity[k], tol);
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
  bulk->mesh_meta_data().use_simple_fields();
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

  double metric_exact[9];
  sierra::nalu::mxm33(Q, Qt, metric_exact);

  const auto& coordField = *static_cast<const VectorFieldType*>(
    bulk->mesh_meta_data().coordinate_field());

  std::vector<DoubleType> ws_coords(topo.num_nodes() * dim);
  const sierra::nalu::SharedMemView<DoubleType**, sierra::nalu::DeviceShmem>
    coords(ws_coords.data(), topo.num_nodes(), dim);
  const auto* nodes = bulk->begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coord = stk::mesh::field_data(coordField, nodes[j]);
    double tmp[3];
    sierra::nalu::matvec33(Q, coord, tmp);
    coords(j, 0) = tmp[0];
    coords(j, 1) = tmp[1];
    coords(j, 2) = tmp[2];
  }

  std::vector<DoubleType> contravariant_metric;
  std::vector<DoubleType> covariant_metric;
  std::tie(contravariant_metric, covariant_metric) =
    calculate_metric_tensor(*mescs, ws_coords);

  for (int ip = 0; ip < mescs->num_integration_points(); ++ip) {
    double identity[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    DoubleType shouldBeIdentity[9];
    sierra::nalu::mxm33(
      &contravariant_metric[9 * ip], &covariant_metric[9 * ip],
      shouldBeIdentity);
    for (unsigned k = 0; k < 9; ++k) {
      EXPECT_NEAR(
        stk::simd::get_data(contravariant_metric[9 * ip + k], 0),
        metric_exact[k], tol);
      EXPECT_NEAR(
        stk::simd::get_data(shouldBeIdentity[k], 0), identity[k], tol);
    }
  }
}

} // namespace

#ifndef KOKKOS_ENABLE_GPU

TEST(MetricTensor, tri3)
{
  test_metric_for_topo_2D(stk::topology::TRIANGLE_3_2D, 1.0e-10);
}

TEST(MetricTensor, quad4)
{
  test_metric_for_topo_2D(stk::topology::QUADRILATERAL_4_2D, 1.0e-10);
}

TEST(MetricTensor, tet4)
{
  test_metric_for_topo_3D(stk::topology::TET_4, 1.0e-10);
}

TEST(MetricTensor, wedge6)
{
  test_metric_for_topo_3D(stk::topology::WEDGE_6, 1.0e-10);
}

TEST(MetricTensor, hex8)
{
  test_metric_for_topo_3D(stk::topology::HEX_8, 1.0e-10);
}

#endif // KOKKOS_ENABLE_GPU
