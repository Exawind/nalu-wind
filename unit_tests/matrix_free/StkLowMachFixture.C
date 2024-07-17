// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkLowMachFixture.h"

#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/LinSysInfo.h"

#include "Tpetra_Map.hpp"

#include "gtest/gtest.h"
#include "mpi.h"
#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/FEMHelpers.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldBase.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/GetEntities.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Selector.hpp"
#include "stk_mesh/base/SkinBoundary.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpField.hpp"
#include "stk_topology/topology.hpp"

#include "stk_unit_test_utils/stk_mesh_fixtures/CoordinateMapping.hpp"
#include "stk_unit_test_utils/stk_mesh_fixtures/HexFixture.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {
LowMachFixture::LowMachFixture(int nx, double scale)
  : bulkPtr(stk::mesh::MeshBuilder(MPI_COMM_WORLD)
              .set_spatial_dimension(3u)
              .set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA)
              .create()),
    meta(bulkPtr->mesh_meta_data()),
    bulk(*bulkPtr),
    io(bulk.parallel()),
    density_field(
      meta.declare_field<double>(stk::topology::NODE_RANK, "density", 3)),
    velocity_field(
      meta.declare_field<double>(stk::topology::NODE_RANK, "velocity", 3)),
    viscosity_field(
      meta.declare_field<double>(stk::topology::NODE_RANK, "viscosity")),
    filter_scale_field(meta.declare_field<double>(
      stk::topology::NODE_RANK, "scaled_filter_length")),
    pressure_field(
      meta.declare_field<double>(stk::topology::NODE_RANK, "pressure")),
    dpdx_field(meta.declare_field<double>(stk::topology::NODE_RANK, "dpdx")),
    dpdx_tmp_field(
      meta.declare_field<double>(stk::topology::NODE_RANK, "dpdx_tmp")),
    body_force_field(
      meta.declare_field<double>(stk::topology::NODE_RANK, "body_force")),
    gid_field(meta.declare_field<gid_type>(
      stk::topology::NODE_RANK, linsys_info::gid_name))
{
  meta.use_simple_fields();
  double one = 1;
  stk::mesh::put_field_on_mesh(gid_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(density_field, meta.universal_part(), &one);
  stk::mesh::put_field_on_mesh(
    velocity_field, meta.universal_part(), 3, nullptr);
  stk::io::set_field_output_type(
    velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::mesh::put_field_on_mesh(viscosity_field, meta.universal_part(), &one);
  stk::mesh::put_field_on_mesh(filter_scale_field, meta.universal_part(), &one);
  stk::mesh::put_field_on_mesh(pressure_field, meta.universal_part(), nullptr);
  stk::mesh::put_field_on_mesh(dpdx_field, meta.universal_part(), 3, nullptr);
  stk::io::set_field_output_type(
    dpdx_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::mesh::put_field_on_mesh(
    dpdx_tmp_field, meta.universal_part(), 3, nullptr);
  stk::io::set_field_output_type(
    dpdx_tmp_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::mesh::put_field_on_mesh(
    body_force_field, meta.universal_part(), 3, nullptr);
  stk::io::set_field_output_type(
    body_force_field, stk::io::FieldOutputType::VECTOR_3D);

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
      const double x = coordptr[0];
      const double y = coordptr[1];
      const double z = coordptr[2];
      coordptr[0] = scale * (x / nx - 0.5);
      coordptr[1] = scale * (y / nx - 0.5);
      coordptr[2] = scale * (z / nx - 0.5);

      const double uInf = 0;
      const double rVortex_ = scale / 8;
      const double beta_ = 0.25;
      const double tFac = 1;
      const double cX = coordptr[0];
      const double cY = coordptr[1];

      const double radius = std::sqrt(cX * cX + cY * cY) / (rVortex_);
      const double factor = beta_ / (rVortex_ * tFac * tFac) *
                            std::exp(0.5 * (1.0 - radius * radius / tFac));

      const double velX = -factor * cY + uInf;
      const double velY = factor * cX;

      stk::mesh::field_data(velocity_field, node)[0] = velX;
      stk::mesh::field_data(velocity_field, node)[1] = velY;
      stk::mesh::field_data(velocity_field, node)[2] = 0;

      for (int d = 0; d < 3; ++d) {
        stk::mesh::field_data(
          velocity_field.field_of_state(stk::mesh::StateN), node)[d] =
          stk::mesh::field_data(velocity_field, node)[d];
      }
    }
  }
  auto vel_ngp = stk::mesh::get_updated_ngp_field<double>(velocity_field);
  vel_ngp.modify_on_host();
  vel_ngp.sync_to_device();

  auto vel_ngp_old = stk::mesh::get_updated_ngp_field<double>(
    velocity_field.field_of_state(stk::mesh::StateN));
  vel_ngp_old.modify_on_host();
  vel_ngp_old.sync_to_device();

  gid_field_ngp = stk::mesh::get_updated_ngp_field<gid_type>(gid_field);
  populate_global_id_field(mesh(), meta.universal_part(), gid_field_ngp);
}
} // namespace matrix_free
} // namespace nalu
} // namespace sierra
