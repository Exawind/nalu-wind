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
#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>

#include <master_element/MasterElement.h>
#include <master_element/MasterElementFunctions.h>
#include <master_element/Hex27CVFEM.h>
#include <master_element/TensorOps.h>

#include <NaluEnv.h>
#include <AlgTraits.h>

#include "UnitTestUtils.h"

namespace {

std::pair<std::vector<double>, std::vector<double>>
calculate_metric_tensor(sierra::nalu::MasterElement& me, const std::vector<double>& ws_coords)
{
  double scs_error = 0.0;
  int gradSize = me.num_integration_points() * me.nodesPerElement_ * me.nDim_;
  std::vector<double> ws_dndx(gradSize);
  std::vector<double> ws_deriv(gradSize);
  std::vector<double> ws_det_j(me.num_integration_points());
  me.grad_op(1, ws_coords.data(), ws_dndx.data(), ws_deriv.data(), ws_det_j.data(), &scs_error);

  int metricSize = me.nDim_ * me.nDim_ * me.num_integration_points();
  std::vector<double> contravariant_metric_tensor(metricSize);
  std::vector<double> covariant_metric_tensor(metricSize);
  me.gij(ws_coords.data(), contravariant_metric_tensor.data(), covariant_metric_tensor.data(), ws_deriv.data());

  return {contravariant_metric_tensor, covariant_metric_tensor};
}

using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;

void test_metric_for_topo_2D(stk::topology topo, double tol) {
  int dim = topo.dimension();
  ASSERT_EQ(dim, 2);

  stk::mesh::MetaData meta(dim);
  stk::mesh::BulkData bulk(meta, MPI_COMM_WORLD);
  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(bulk, topo);

  auto* mescs = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  // apply some arbitrary linear map the reference element
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double Q[4] = { 1.0+std::abs(coeff(rng)), coeff(rng), coeff(rng), 1.0+std::abs(coeff(rng)) };

  double Qt[4];
  sierra::nalu::transpose22(Q, Qt);

  double metric_exact[4];
  sierra::nalu::mxm22(Q,Qt,metric_exact);


  const auto& coordField = *static_cast<const VectorFieldType*>(meta.coordinate_field());
  std::vector<double> ws_coords(topo.num_nodes() * dim);
  const auto* nodes = bulk.begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, nodes[j]);
    sierra::nalu::matvec22(Q, coords, &ws_coords[j*dim]);
  }

  std::vector<double> contravariant_metric; std::vector<double> covariant_metric;
  std::tie(contravariant_metric, covariant_metric) = calculate_metric_tensor(*mescs, ws_coords);

  for (int ip = 0; ip < mescs->num_integration_points(); ++ip) {
    double identity[4] = {1.0,0.0,0.0,1.0};
    double shouldBeIdentity[4];
    sierra::nalu::mxm22(&contravariant_metric[4*ip], &covariant_metric[4*ip], shouldBeIdentity);
    for (unsigned k = 0; k < 4; ++k) {
      EXPECT_NEAR(contravariant_metric[4*ip+k], metric_exact[k], tol);
      EXPECT_NEAR(shouldBeIdentity[k], identity[k], tol);
    }
  }
}

void test_metric_for_topo_3D(stk::topology topo, double tol) {
  int dim = topo.dimension();
  ASSERT_EQ(dim,3);

  stk::mesh::MetaData meta(dim);
  stk::mesh::BulkData bulk(meta, MPI_COMM_WORLD);
  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(bulk, topo);

  auto* mescs = sierra::nalu::MasterElementRepo::get_surface_master_element(topo);

  // apply some arbitrary linear map the reference element
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double Q[9] = {
      1.0+std::abs(coeff(rng)), coeff(rng), coeff(rng),
      coeff(rng), 1.0+std::abs(coeff(rng)), coeff(rng),
      coeff(rng), coeff(rng), 1.0+std::abs(coeff(rng))
  };

  double Qt[9];
  sierra::nalu::transpose33(Q,Qt);

  double metric_exact[9];
  sierra::nalu::mxm33(Q,Qt,metric_exact);


  const auto& coordField = *static_cast<const VectorFieldType*>(meta.coordinate_field());

  std::vector<double> ws_coords(topo.num_nodes() * dim);
  const auto* nodes = bulk.begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, nodes[j]);
    sierra::nalu::matvec33(Q, coords, &ws_coords[j*dim]);
  }

  std::vector<double> contravariant_metric; std::vector<double> covariant_metric;
  std::tie(contravariant_metric, covariant_metric) = calculate_metric_tensor(*mescs, ws_coords);

  for (int ip = 0; ip < mescs->num_integration_points(); ++ip) {
    double identity[9] = {
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0
    };

    double shouldBeIdentity[9];
    sierra::nalu::mxm33(&contravariant_metric[9*ip], &covariant_metric[9*ip], shouldBeIdentity);
    for (unsigned k = 0; k < 9; ++k) {
      EXPECT_NEAR(contravariant_metric[9*ip+k], metric_exact[k], tol);
      EXPECT_NEAR(shouldBeIdentity[k], identity[k], tol);
    }
  }
}

}

TEST(MetricTensor, tri3)
{
  test_metric_for_topo_2D(stk::topology::TRIANGLE_3_2D, 1.0e-10);
}

TEST(MetricTensor, quad4)
{
  test_metric_for_topo_2D(stk::topology::QUADRILATERAL_4_2D,1.0e-10);
}

TEST(MetricTensor, quad9)
{
  test_metric_for_topo_2D(stk::topology::QUADRILATERAL_9_2D,1.0e-10);
}

TEST(MetricTensor, tet4)
{
  test_metric_for_topo_3D(stk::topology::TET_4,1.0e-10);
}

TEST(MetricTensor, wedge6)
{
  test_metric_for_topo_3D(stk::topology::WEDGE_6,1.0e-10);
}

TEST(MetricTensor, hex8)
{
  test_metric_for_topo_3D(stk::topology::HEX_8,1.0e-10);
}
TEST(MetricTensor, hex27)
{
  test_metric_for_topo_3D(stk::topology::HEX_27,1.0e-10);
}


TEST(MetricTensorNGP, hex27)
{
  stk::topology topo =  stk::topology::HEX_27;
  int dim = topo.dimension();
  ASSERT_EQ(dim,3);

  stk::mesh::MetaData meta(dim);
  stk::mesh::BulkData bulk(meta, MPI_COMM_WORLD);
  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(bulk, topo);

  sierra::nalu::Hex27SCS mescs;

  // apply some arbitrary linear map the reference element
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);

  double Q[9] = {
      1.0+std::abs(coeff(rng)), coeff(rng), coeff(rng),
      coeff(rng), 1.0+std::abs(coeff(rng)), coeff(rng),
      coeff(rng), coeff(rng), 1.0+std::abs(coeff(rng))
  };

  double Qt[9];
  sierra::nalu::transpose33(Q,Qt);

  double metric_exact[9];
  sierra::nalu::mxm33(Q,Qt,metric_exact);

  const auto& coordField = *static_cast<const VectorFieldType*>(meta.coordinate_field());

  Kokkos::View<double**> v_coords("coords", 27, 3);
  const auto* nodes = bulk.begin_nodes(elem);
  for (unsigned j = 0; j < topo.num_nodes(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, nodes[j]);
    sierra::nalu::matvec33(Q, coords, &v_coords(j,0));
  }

  Kokkos::View<double***> gUpper("gupper", 216, dim, dim);
  Kokkos::View<double***> gLower("glower", 216, dim, dim);

  using AlgTraits = sierra::nalu::AlgTraitsHex27;
  using GradViewType = Kokkos::View<double[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_][AlgTraits::nDim_]>;
  GradViewType refGrad = mescs.copy_deriv_weights_to_view<GradViewType>();

  sierra::nalu::generic_gij_3d<sierra::nalu::AlgTraitsHex27>(refGrad, v_coords, gUpper, gLower);

  for (int ip = 0; ip < mescs.num_integration_points(); ++ip) {
    double identity[9] = {
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0
    };

    double shouldBeIdentity[9];
    sierra::nalu::mxm33(&gUpper(ip,0,0), &gLower(ip,0,0), shouldBeIdentity);
    for (int d_outer = 0; d_outer < 3; ++d_outer) {
      for (int d_inner = 0; d_inner < 3; ++d_inner) {
        EXPECT_NEAR(gUpper(ip, d_outer, d_inner), metric_exact[3*d_outer+d_inner], tol);
        EXPECT_NEAR(shouldBeIdentity[3*d_outer+d_inner], identity[3*d_outer+d_inner], tol);
      }
    }
  }
}
