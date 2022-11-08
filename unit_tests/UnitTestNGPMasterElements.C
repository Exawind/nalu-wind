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
#include <master_element/Hex27CVFEM.h>
#include <master_element/Tri32DCVFEM.h>
#include <AlgTraits.h>

#include <memory>
#include <random>
#include <chrono>

#include <UnitTestUtils.h>

using clock_type = std::chrono::steady_clock;
using VectorFieldType = stk::mesh::Field<double, stk::mesh::Cartesian>;
//-------------------------------------------------------------------------

namespace {

double
linear_scalar_value(int dim, double a, const double* b, const double* x)
{
  if (dim == 2) {
    return (a + b[0] * x[0] + b[1] * x[1]);
  }
  return (a + b[0] * x[0] + b[1] * x[1] + b[2] * x[2]);
}

} // namespace

#if !defined(KOKKOS_ENABLE_GPU)
TEST(MasterElementFunctions, generic_grad_op_3d_hex_27)
{
  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(3);
  auto bulk = meshBuilder.create();

  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(
    *bulk, stk::topology::HEXAHEDRON_27);
  const auto* node_rels = bulk->begin_nodes(elem);
  sierra::nalu::Hex27SCS me;
  auto& coordField = *static_cast<const VectorFieldType*>(
    bulk->mesh_meta_data().coordinate_field());
  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  double a = coeff(rng);
  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }

  std::vector<double> polyResult(me.num_integration_points() * dim);
  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (int d = 0; d < dim; ++d) {
      polyResult[j * dim + d] = coeffs[d];
    }
  }

  std::vector<double> ws_field(me.nodes_per_element());
  Kokkos::View<double**> ws_coords("coords", me.nodes_per_element(), dim);
  for (int j = 0; j < me.nodes_per_element(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords(j, d) = coords[d];
    }
    ws_field[j] = linear_scalar_value(dim, a, coeffs.data(), coords);
  }

  Kokkos::View<double***> meGrad(
    "grad", me.num_integration_points(), me.nodes_per_element(), dim);

  using AlgTraits = sierra::nalu::AlgTraitsHex27;
  using GradViewType =
    Kokkos::View<double[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]
                       [AlgTraits::nDim_]>;
  GradViewType refGrad = me.copy_deriv_weights_to_view<GradViewType>();

  double duration = 0;

#ifndef NDEBUG
  int nIt = 10000;
#else
  int nIt = 1;
#endif

  for (int k = 0; k < nIt; ++k) {
    Kokkos::deep_copy(meGrad, 0.0);
    auto start_clock = clock_type::now();
    sierra::nalu::generic_grad_op<AlgTraits>(refGrad, ws_coords, meGrad);
    auto end_clock = clock_type::now();
    duration += 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_clock - start_clock)
                           .count();
  }
  std::cout << "Time per iteration: " << (duration / nIt) * 1000 << "(ms)"
            << std::endl;

  std::vector<double> meResult(me.num_integration_points() * dim, 0.0);
  for (int ip = 0; ip < me.num_integration_points(); ++ip) {
    for (int n = 0; n < me.nodes_per_element(); ++n) {
      for (int d = 0; d < dim; ++d) {
        meResult[ip * dim + d] += meGrad(ip, n, d) * ws_field[n];
      }
    }
  }

  // derivative should be exact to floating point error
  for (unsigned j = 0; j < meResult.size(); ++j) {
    EXPECT_NEAR(meResult[j], polyResult[j], tol);
  }
}
#endif

TEST(MasterElementFunctions, generic_grad_op_2d_tri_6)
{
  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(2);
  auto bulk = meshBuilder.create();

  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(
    *bulk, stk::topology::TRIANGLE_3_2D);
  const auto* node_rels = bulk->begin_nodes(elem);
  sierra::nalu::Tri32DSCS me;
  auto& coordField = *static_cast<const VectorFieldType*>(
    bulk->mesh_meta_data().coordinate_field());
  int dim = me.ndim();

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  double a = coeff(rng);
  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }

  std::vector<double> polyResult(me.num_integration_points() * dim);
  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (int d = 0; d < dim; ++d) {
      polyResult[j * dim + d] = coeffs[d];
    }
  }

  std::vector<double> ws_field(me.nodes_per_element());
  Kokkos::View<double**> ws_coords("coords", me.nodes_per_element(), dim);
  for (int j = 0; j < me.nodes_per_element(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords(j, d) = coords[d];
    }
    ws_field[j] = linear_scalar_value(dim, a, coeffs.data(), coords);
  }

  Kokkos::View<double***> meGrad(
    "grad", me.num_integration_points(), me.nodes_per_element(), dim);

  using AlgTraits = sierra::nalu::AlgTraitsTri3_2D;

  Kokkos::View<double***> refGrad(
    "reference_gradient_weights", me.num_integration_points(),
    me.nodes_per_element(), dim);
  for (int j = 0; j < me.num_integration_points(); ++j) {
    refGrad(j, 0, 0) = -1.0;
    refGrad(j, 1, 0) = 1.0;
    refGrad(j, 2, 0) = 0.0;
    refGrad(j, 0, 1) = -1.0;
    refGrad(j, 1, 1) = 0.0;
    refGrad(j, 2, 1) = 1.0;
  }

  double duration = 0;
#ifndef NDEBUG
  int nIt = 10000;
#else
  int nIt = 1;
#endif
  for (int k = 0; k < nIt; ++k) {
    Kokkos::deep_copy(meGrad, 0.0);
    auto start_clock = clock_type::now();
    sierra::nalu::generic_grad_op<AlgTraits>(refGrad, ws_coords, meGrad);
    auto end_clock = clock_type::now();
    duration += 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_clock - start_clock)
                           .count();
  }
  std::cout << "Time per iteration: " << (duration / nIt) * 1000 << "(ms)"
            << std::endl;

  std::vector<double> meResult(me.num_integration_points() * dim, 0.0);
  for (int ip = 0; ip < me.num_integration_points(); ++ip) {
    for (int n = 0; n < me.nodes_per_element(); ++n) {
      for (int d = 0; d < dim; ++d) {
        meResult[ip * dim + d] += meGrad(ip, n, d) * ws_field[n];
      }
    }
  }

  // derivative should be exact to floating point error
  for (unsigned j = 0; j < meResult.size(); ++j) {
    EXPECT_NEAR(meResult[j], polyResult[j], tol);
  }
}

#if !defined(KOKKOS_ENABLE_GPU)
TEST(Hex27SCV, detj)
{
  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(3);
  auto bulk = meshBuilder.create();

  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(
    *bulk, stk::topology::HEXAHEDRON_27);
  const auto* node_rels = bulk->begin_nodes(elem);
  sierra::nalu::Hex27SCV me;

  auto& coordField = *static_cast<const VectorFieldType*>(
    bulk->mesh_meta_data().coordinate_field());

  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }

  std::vector<double> polyResult(me.num_integration_points() * dim);
  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (int d = 0; d < dim; ++d) {
      polyResult[j * dim + d] = coeffs[d];
    }
  }

  Kokkos::View<double**> ws_coords("coords", me.nodes_per_element(), dim);
  for (int j = 0; j < me.nodes_per_element(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords(j, d) = coords[d];
    }
  }

  Kokkos::View<double*> meDetj("detj", me.num_integration_points());

  using AlgTraits = sierra::nalu::AlgTraitsHex27;
  using GradViewType =
    Kokkos::View<double[AlgTraits::numScvIp_][AlgTraits::nodesPerElement_]
                       [AlgTraits::nDim_]>;
  GradViewType refGrad = me.copy_deriv_weights_to_view<GradViewType>();

  double duration = 0;
#ifndef NDEBUG
  int nIt = 10000;
#else
  int nIt = 1;
#endif
  for (int k = 0; k < nIt; ++k) {
    Kokkos::deep_copy(meDetj, 0.0);
    auto start_clock = clock_type::now();
    me.weighted_volumes(refGrad, ws_coords, meDetj);
    auto end_clock = clock_type::now();
    duration += 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_clock - start_clock)
                           .count();
  }
  std::cout << "Time per iteration: " << (duration / nIt) * 1000 << "(ms)"
            << std::endl;

  constexpr int nTypes = 4;
  int typeCount[nTypes] = {0, 0, 0, 0};
  double exactVolumeType[4];

  double cornerType = 1 - std::sqrt(1.0 / 3.0);
  double centerType = 2.0 * std::sqrt(1.0 / 3.0);

  exactVolumeType[0] = 0.125 * cornerType * cornerType * cornerType;
  exactVolumeType[1] = 0.125 * cornerType * cornerType * centerType;
  exactVolumeType[2] = 0.125 * cornerType * centerType * centerType;
  exactVolumeType[3] = 0.125 * centerType * centerType * centerType;

  for (int ip = 0; ip < me.num_integration_points(); ++ip) {
    EXPECT_GT(meDetj(ip), tol);
    for (int i = 0; i < nTypes; ++i) {
      if (std::abs(exactVolumeType[i] - meDetj(ip)) < tol) {
        ++typeCount[i];
      }
    }
  }

  EXPECT_EQ(typeCount[0], 8 * 8);
  EXPECT_EQ(typeCount[1], 8 * 12);
  EXPECT_EQ(typeCount[2], 8 * 6);
  EXPECT_EQ(typeCount[3], 8 * 1);
}

TEST(Hex27SCS, area_vec)
{
  stk::mesh::MeshBuilder meshBuilder(MPI_COMM_WORLD);
  meshBuilder.set_spatial_dimension(3);
  auto bulk = meshBuilder.create();
  auto& meta = bulk->mesh_meta_data();

  stk::mesh::Entity elem = unit_test_utils::create_one_reference_element(
    *bulk, stk::topology::HEXAHEDRON_27);
  const auto* node_rels = bulk->begin_nodes(elem);
  sierra::nalu::Hex27SCS me;
  auto& coordField =
    *static_cast<const VectorFieldType*>(meta.coordinate_field());
  int dim = me.nDim_;

  std::mt19937 rng;
  rng.seed(std::mt19937::default_seed);
  std::uniform_real_distribution<double> coeff(-1.0, 1.0);
  std::vector<double> coeffs(dim);

  for (int j = 0; j < dim; ++j) {
    coeffs[j] = coeff(rng);
  }

  std::vector<double> polyResult(me.num_integration_points() * dim);
  for (int j = 0; j < me.num_integration_points(); ++j) {
    for (int d = 0; d < dim; ++d) {
      polyResult[j * dim + d] = coeffs[d];
    }
  }

  Kokkos::View<double**> ws_coords("coords", me.nodes_per_element(), dim);
  for (int j = 0; j < me.nodes_per_element(); ++j) {
    const double* coords = stk::mesh::field_data(coordField, node_rels[j]);
    for (int d = 0; d < dim; ++d) {
      ws_coords(j, d) = coords[d];
    }
  }

  Kokkos::View<double**> meAreav("area_vec", me.num_integration_points(), dim);

  using AlgTraits = sierra::nalu::AlgTraitsHex27;
  using GradViewType =
    Kokkos::View<double[AlgTraits::numScsIp_][AlgTraits::nodesPerElement_]
                       [AlgTraits::nDim_]>;
  GradViewType refGrad = me.copy_deriv_weights_to_view<GradViewType>();

  double duration = 0;
  int nIt = 10000;
  for (int k = 0; k < nIt; ++k) {
    Kokkos::deep_copy(meAreav, 0.0);
    auto start_clock = clock_type::now();
    me.weighted_area_vectors(refGrad, ws_coords, meAreav);
    auto end_clock = clock_type::now();
    duration += 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_clock - start_clock)
                           .count();
  }
  std::cout << "Time per iteration: " << (duration / nIt) * 1000 << "(ms)"
            << std::endl;

  constexpr int nTypes = 3;
  int typeCount[nTypes] = {0, 0, 0};
  double exactAreaType[nTypes];

  exactAreaType[0] = 0.25 * (1 - std::sqrt(1. / 3.)) * (1 - std::sqrt(1. / 3.));
  exactAreaType[1] = 1.0 / 6.0 * (std::sqrt(3.) - 1);
  exactAreaType[2] = 1.0 / 3.0;

  for (int ip = 0; ip < me.num_integration_points(); ++ip) {
    double mag = meAreav(ip, 0) * meAreav(ip, 0) +
                 meAreav(ip, 1) * meAreav(ip, 1) +
                 meAreav(ip, 2) * meAreav(ip, 2);
    EXPECT_GT(mag, tol);

    for (int i = 0; i < nTypes; ++i) {
      const double Asq = exactAreaType[i] * exactAreaType[i];
      if (std::abs(Asq - mag) < tol) {
        ++typeCount[i];
      }
    }
  }

  EXPECT_EQ(typeCount[0], 96);
  EXPECT_EQ(typeCount[1], 96);
  EXPECT_EQ(typeCount[2], 24);
}
#endif
