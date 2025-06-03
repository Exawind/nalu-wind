// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "StkConductionFixture.h"
#include "matrix_free/ConductionInfo.h"
#include "matrix_free/StkToTpetraMap.h"
#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_topology/topology.hpp"
#include "stk_io/DatabasePurpose.hpp"
#include "stk_io/IossBridge.hpp"
#include "stk_mesh/base/Bucket.hpp"
#include "stk_mesh/base/FieldState.hpp"
#include "stk_mesh/base/Ngp.hpp"
#include "stk_mesh/base/Types.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/Hex27Fixture.hpp"

#include <iosfwd>
#include <string>

ConductionFixture::ConductionFixture(int nx, double scale)
  : bulkPtr(
      stk::mesh::MeshBuilder(MPI_COMM_WORLD)
        .set_spatial_dimension(3u)
        .set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA)
        .create()),
    meta(bulkPtr->mesh_meta_data()),
    bulk(*bulkPtr),
    io(bulk.parallel()),
    q_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::q_name,
      3)),
    qbc_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::qbc_name)),
    flux_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::flux_name)),
    qtmp_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::qtmp_name)),
    alpha_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::volume_weight_name)),
    lambda_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::diffusion_weight_name)),
    gid_field(meta.declare_field<typename Tpetra::Map<>::global_ordinal_type>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::gid_name))
{
  meta.use_simple_fields();
  stk::mesh::put_field_on_mesh(gid_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(q_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(qbc_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(flux_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(qtmp_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(alpha_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(lambda_field, meta.universal_part(), nullptr);

  const std::string nx_s = std::to_string(nx);
  const std::string name =
    "generated:" + nx_s + "x" + nx_s + "x" + nx_s + "|sideset:xXyYzZ";
  io.set_bulk_data(bulk);
  io.add_mesh_database(name, stk::io::READ_MESH);
  io.create_input_mesh();
  io.populate_bulk_data();
  stk::io::put_io_part_attribute(meta.universal_part());

  auto& coord_field = coordinate_field();
  for (auto ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      auto* coordptr = stk::mesh::field_data(coord_field, node);
      coordptr[0] = scale * (coordptr[0] / nx - 0.5);
      coordptr[1] = scale * (coordptr[1] / nx - 0.5);
      coordptr[2] = scale * (coordptr[2] / nx - 0.5);
    }
  }

  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNP1), node) = 1.0;
      *stk::mesh::field_data(q_field.field_of_state(stk::mesh::StateN), node) =
        1.0;
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNM1), node) = 1.0;
      *stk::mesh::field_data(alpha_field, node) = 1.0;
      *stk::mesh::field_data(lambda_field, node) = 1.0;
      *stk::mesh::field_data(gid_field, node) = bulk.identifier(node);
    }
  }
  mesh = stk::mesh::get_updated_ngp_mesh(bulk);
  gid_field_ngp = stk::mesh::get_updated_ngp_field<gid_type>(gid_field);
  sierra::nalu::matrix_free::populate_global_id_field(
    mesh, meta.universal_part(), gid_field_ngp);
}

stk::mesh::Field<double>&
ConductionFixture::coordinate_field()
{
  return *meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
}

ConductionFixtureP2::ConductionFixtureP2(int nx, double scale)
  : fixture(MPI_COMM_WORLD, nx, nx, nx, stk::mesh::BulkData::NO_AUTO_AURA),
    meta(fixture.m_meta),
    bulk(fixture.m_bulk_data),
    io(bulk.parallel()),
    q_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::q_name,
      3)),
    qtmp_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::qtmp_name,
      3)),
    alpha_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::volume_weight_name)),
    lambda_field(meta.declare_field<double>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::diffusion_weight_name)),
    gid_field(meta.declare_field<typename Tpetra::Map<>::global_ordinal_type>(
      stk::topology::NODE_RANK,
      sierra::nalu::matrix_free::conduction_info::gid_name))
{
  for (auto* part : fixture.m_elem_parts) {
    stk::io::put_io_part_attribute(*part);
  }

  stk::mesh::put_field_on_mesh(gid_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(q_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(qtmp_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(alpha_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(lambda_field, meta.universal_part(), nullptr);
  fixture.generate_mesh();

  auto& coordField = coordinate_field();
  for (auto ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      auto* coordptr = stk::mesh::field_data(coordField, node);
      coordptr[0] = scale * (coordptr[0] / (2 * nx) - 0.5);
      coordptr[1] = scale * (coordptr[1] / (2 * nx) - 0.5);
      coordptr[2] = scale * (coordptr[2] / (2 * nx) - 0.5);
    }
  }

  for (const auto* ib :
       bulk.get_buckets(stk::topology::NODE_RANK, meta.universal_part())) {
    for (auto node : *ib) {
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNP1), node) = 1.0;
      *stk::mesh::field_data(q_field.field_of_state(stk::mesh::StateN), node) =
        1.0;
      *stk::mesh::field_data(
        q_field.field_of_state(stk::mesh::StateNM1), node) = 1.0;
      *stk::mesh::field_data(alpha_field, node) = 1.0;
      *stk::mesh::field_data(lambda_field, node) = 1.0;
      *stk::mesh::field_data(gid_field, node) = bulk.identifier(node);
    }
  }
  io.set_bulk_data(bulk);
  mesh = stk::mesh::get_updated_ngp_mesh(bulk);
  gid_field_ngp = stk::mesh::get_updated_ngp_field<gid_type>(gid_field);
  sierra::nalu::matrix_free::populate_global_id_field(
    mesh, meta.universal_part(), gid_field_ngp);
}

stk::mesh::Field<double>&
ConductionFixtureP2::coordinate_field()
{
  return *meta.get_field<double>(stk::topology::NODE_RANK, "coordinates");
}
