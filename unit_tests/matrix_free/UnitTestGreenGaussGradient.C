// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/GradientSolutionUpdate.h"
#include "matrix_free/GreenGaussGradient.h"
#include "matrix_free/GreenGaussGradientOperator.h"
#include "matrix_free/LinearAreas.h"
#include "matrix_free/LinearVolume.h"
#include "matrix_free/StkGradientFixture.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkToTpetraMap.h"

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Tpetra_Export.hpp"
#include "Tpetra_MultiVector.hpp"

#include "stk_io/DatabasePurpose.hpp"
#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/GetNgpField.hpp"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_topology/topology.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include <iosfwd>
#include <random>
#include <math.h>
#include <type_traits>
#include <vector>

namespace sierra {
namespace nalu {
namespace matrix_free {

class GradientSolveFixture : public GradientFixture
{
protected:
  GradientSolveFixture()
    : GradientFixture(nx, M_PI),
      linsys(mesh(), active(), gid_field_ngp),
      exporter(
        Teuchos::rcpFromRef(linsys.owned_and_shared),
        Teuchos::rcpFromRef(linsys.owned)),
      offsets(create_offset_map<order>(
        mesh(), active(), linsys.stk_lid_to_tpetra_lid)),
      bc_faces(
        face_offsets<order>(mesh(), side(), linsys.stk_lid_to_tpetra_lid))
  {
    for (auto ib :
         bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
      for (auto node : *ib) {
        const auto* cx = stk::mesh::field_data(coordinate_field(), node);
        *stk::mesh::field_data(q_field, node) = func(cx[0], cx[1], cx[2]);
        for (int d = 0; d < 3; ++d) {
          stk::mesh::field_data(dqdx_exact_field, node)[d] =
            grad(d, cx[0], cx[1], cx[2]);
        }
      }
    }
  }

  const double kx = 1;
  const double ky = 1;
  const double kz = 1;

  double func(double x, double y, double z)
  {
    return std::cos(kx * x) * std::cos(ky * y) * std::cos(kz * z);
  }

  double grad(int d, double x, double y, double z)
  {
    if (d == 0) {
      return -kx * std::sin(kx * x) * std::cos(ky * y) * std::cos(kz * z);
    } else if (d == 1) {
      return -ky * std::cos(kx * x) * std::sin(ky * y) * std::cos(kz * z);
    } else {
      return -kz * std::cos(kx * x) * std::cos(ky * y) * std::sin(kz * z);
    }
  }

  double error()
  {
    double err = 0;
    int count = 0;
    for (auto ib : bulk.get_buckets(
           stk::topology::NODE_RANK, meta.locally_owned_part())) {
      for (auto node : *ib) {
        const auto* cx = stk::mesh::field_data(coordinate_field(), node);

        for (int d = 0; d < 3; ++d) {
          double lerr = stk::mesh::field_data(dqdx_field, node)[d] -
                        grad(d, cx[0], cx[1], cx[2]);
          err += lerr * lerr;
        }
        ++count;
      }
    }
    double g_err;
    stk::all_reduce_sum(bulk.parallel(), &err, &g_err, 1);
    int g_count;
    stk::all_reduce_sum(bulk.parallel(), &count, &g_count, 1);
    return std::sqrt(g_err / g_count);
  }

  GradientResidualFields<order> gather_required_fields()
  {
    auto conn = stk_connectivity_map<order>(mesh(), meta.universal_part());
    GradientResidualFields<order> fields;
    fields.q = scalar_view<order>("q", offsets.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(q_field), fields.q);

    fields.dqdx = vector_view<order>("dqdx", offsets.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(dqdx_field), fields.dqdx);

    auto coords = vector_view<order>("coords", conn.extent_int(0));
    field_gather<order>(
      conn, stk::mesh::get_updated_ngp_field<double>(coordinate_field()),
      coords);

    fields.vols = geom::volume_metric<order>(coords);
    fields.areas = geom::linear_areas<order>(coords);

    return fields;
  }

  Teuchos::ParameterList params{};
  StkToTpetraMaps linsys;
  Tpetra::Export<> exporter;
  const_elem_offset_view<order> offsets;
  face_offset_view<order> bc_faces;

  static constexpr int nx = 32;
  static constexpr double scale = M_PI;
  static constexpr double k = 0.5 * M_PI / scale;
};

TEST_F(GradientSolveFixture, create)
{
  ASSERT_NO_THROW(
    GradientSolutionUpdate<order>(params, linsys, exporter, offsets, bc_faces));
}

TEST_F(GradientSolveFixture, residual_is_greater_than_zero)
{
  GradientSolutionUpdate<order> update(params, linsys, exporter, offsets, {});
  update.compute_residual(gather_required_fields(), {});
  ASSERT_GT(update.residual_norm(), 0);
}

TEST_F(GradientSolveFixture, solve_is_reasonable)
{
  bc_faces = face_offset_view<order>("d", 0);
  GradientSolutionUpdate<order> update(
    params, linsys, exporter, offsets, bc_faces);

  auto fields = gather_required_fields();
  update.compute_residual(fields, {});
  auto& delta = update.compute_delta(fields.vols);

  const int num_vectors(delta.getNumVectors());
  Teuchos::Array<double> mv_norm(num_vectors);
  delta.norm2(mv_norm());

  double norm = 0;
  for (int k = 0; k < num_vectors; ++k) {
    norm += mv_norm[k] * mv_norm[k];
  }
  norm = std::sqrt(norm);

  ASSERT_GT(norm, 0);
  ASSERT_GT(update.num_iterations(), 1);
  ASSERT_LT(update.num_iterations(), 100);
}

void
dump_mesh(
  stk::mesh::BulkData& bulk,
  std::vector<stk::mesh::FieldBase*> fields,
  std::string name)
{
  stk::io::StkMeshIoBroker io(bulk.parallel());
  io.set_bulk_data(bulk);
  auto fileId = io.create_output_mesh(name, stk::io::WRITE_RESULTS);

  for (auto* field : fields) {
    io.add_field(fileId, *field);
  }
  io.process_output_request(fileId, 0.0);
}

class ComputeGradientFixture : public GradientSolveFixture
{
protected:
  ComputeGradientFixture()
    : GradientSolveFixture(),
      conn(stk_connectivity_map<order>(mesh(), active())),
      face_conn(face_node_map<order>(mesh(), side())),
      grad(params, meta, linsys, exporter, conn, offsets, face_conn, bc_faces)
  {
  }

  const_elem_mesh_index_view<order> conn;
  const_face_mesh_index_view<order> face_conn;
  ComputeGradient<order> grad;
};

TEST_F(ComputeGradientFixture, create) {}

TEST_F(ComputeGradientFixture, correct_behavior_for_linear_field)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-2.0, 2.0);
  for (auto ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      const auto* cx = stk::mesh::field_data(coordinate_field(), node);
      *stk::mesh::field_data(q_field, node) =
        0.3 + 2.2 * cx[0] + 0.9 * cx[1] - 4.3 * cx[2];
      stk::mesh::field_data(dqdx_field, node)[0] = coeff(rng);
      stk::mesh::field_data(dqdx_field, node)[1] = coeff(rng);
      stk::mesh::field_data(dqdx_field, node)[2] = coeff(rng);
    }
  }
  auto& q = stk::mesh::get_updated_ngp_field<double>(q_field);
  auto& gq = stk::mesh::get_updated_ngp_field<double>(dqdx_field);
  grad.gradient(mesh(), active(), q, gq);
  gq.sync_to_host();
  for (auto ib : bulk.get_buckets(stk::topology::NODE_RANK, active())) {
    for (auto node : *ib) {
      ASSERT_NEAR(stk::mesh::field_data(dqdx_field, node)[0], +2.2, 1.0e-3);
      ASSERT_NEAR(stk::mesh::field_data(dqdx_field, node)[1], +0.9, 1.0e-3);
      ASSERT_NEAR(stk::mesh::field_data(dqdx_field, node)[2], -4.3, 1.0e-3);
    }
  }
}

TEST_F(ComputeGradientFixture, error_in_gradient_is_smallish_for_harmonic_field)
{
  std::mt19937 rng;
  rng.seed(0); // fixed seed
  std::uniform_real_distribution<double> coeff(-2.0, 2.0);
  for (auto ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      stk::mesh::field_data(dqdx_field, node)[0] = coeff(rng);
      stk::mesh::field_data(dqdx_field, node)[1] = coeff(rng);
      stk::mesh::field_data(dqdx_field, node)[2] = coeff(rng);
    }
  }

  GreenGaussGradient<1> g_grad(
    bulk, Teuchos::ParameterList{}, meta.universal_part(), side(),
    stk::mesh::Selector{});

  const auto& q = stk::mesh::get_updated_ngp_field<double>(q_field);
  auto& gq = stk::mesh::get_updated_ngp_field<double>(dqdx_field);
  g_grad.gradient(q, gq);
  gq.sync_to_host();
  std::string fileName("test_data.e");
  dump_mesh(bulk, {&q_field, &dqdx_field, &dqdx_exact_field}, fileName);
  ASSERT_LT(error(), 5e-3);
  unlink(fileName.c_str());
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
