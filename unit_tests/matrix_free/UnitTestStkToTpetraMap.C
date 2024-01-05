// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/StkToTpetraMap.h"
#include "matrix_free/StkToTpetraLocalIndices.h"

#include "gtest/gtest.h"
#include "Tpetra_Map.hpp"
#include "stk_io/StkMeshIoBroker.hpp"
#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/MeshBuilder.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/GetNgpMesh.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

namespace stk {
namespace mesh {
class Part;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {
namespace matrix_free {

class StkMeshFixture : public ::testing::Test
{
protected:
  StkMeshFixture()
    : bulkPtr(stk::mesh::MeshBuilder(MPI_COMM_WORLD)
                .set_spatial_dimension(3u)
                .set_aura_option(stk::mesh::BulkData::NO_AUTO_AURA)
                .create()),
      bulk(*bulkPtr),
      meta(bulk.mesh_meta_data()),
      gid_field_h(
        meta.declare_field<
          stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>>(
          stk::topology::NODE_RANK, "global_ids"))
  {
    active = meta.locally_owned_part() | meta.globally_shared_part();
    stk::mesh::put_field_on_mesh(gid_field_h, active, 1, nullptr);
    stk::io::StkMeshIoBroker io(bulk.parallel());
    const auto num_procs = std::to_string(bulk.parallel_size());
    const auto name = "generated:1x1x" + num_procs;
    io.set_bulk_data(bulk);
    io.add_mesh_database(name, stk::io::READ_MESH);
    io.create_input_mesh();
    io.populate_bulk_data();
    mesh = stk::mesh::get_updated_ngp_mesh(bulk);
    using gid_type = typename Tpetra::Map<>::global_ordinal_type;
    gid_field = stk::mesh::get_updated_ngp_field<gid_type>(gid_field_h);
  }

  std::shared_ptr<stk::mesh::BulkData> bulkPtr;
  stk::mesh::BulkData& bulk;
  stk::mesh::MetaData& meta;
  stk::mesh::Field<typename Tpetra::Map<>::global_ordinal_type>& gid_field_h;
  stk::mesh::NgpField<typename Tpetra::Map<>::global_ordinal_type> gid_field;
  stk::mesh::NgpMesh mesh;
  stk::mesh::Selector active;
};

class GlobalIDFieldFixture : public StkMeshFixture
{
};

TEST_F(GlobalIDFieldFixture, populate_global_ids)
{
  populate_global_id_field(mesh, active, gid_field);
}

TEST_F(GlobalIDFieldFixture, field_global_ids_are_unique)
{
  populate_global_id_field(mesh, active, gid_field);
  gid_field.sync_to_host();

  std::set<typename Tpetra::Map<>::global_ordinal_type> idxs;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto node : *ib) {
      auto it = idxs.insert(*stk::mesh::field_data(gid_field_h, node));
      ASSERT_TRUE(it.second);
    }
  }
}

TEST_F(GlobalIDFieldFixture, global_id_field_has_correct_number_of_unique_ids)
{
  populate_global_id_field(mesh, active, gid_field);
  gid_field.sync_to_host();

  std::set<typename Tpetra::Map<>::global_ordinal_type> idxs;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto node : *ib) {
      idxs.insert(*stk::mesh::field_data(gid_field_h, node));
    }
  }
  ASSERT_EQ(idxs.size(), 8u);
}

TEST_F(GlobalIDFieldFixture, global_id_field_has_correct_maximum)
{
  populate_global_id_field(mesh, active, gid_field);
  gid_field.sync_to_host();

  typename Tpetra::Map<>::global_ordinal_type local_max_id = -1;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto node : *ib) {
      local_max_id =
        std::max(local_max_id, *stk::mesh::field_data(gid_field_h, node));
    }
  }

  typename Tpetra::Map<>::global_ordinal_type global_max_id = -1;
  stk::all_reduce_max(bulk.parallel(), &local_max_id, &global_max_id, 1u);

  ASSERT_EQ(global_max_id, 8 + (bulk.parallel_size() - 1) * 4);
}

class MapFixture : public StkMeshFixture
{
protected:
  MapFixture() : StkMeshFixture()
  {
    populate_global_id_field(mesh, active, gid_field);
  }
};

TEST_F(MapFixture, successful_owned_map_creation)
{
  ASSERT_NO_THROW(make_owned_row_map(mesh, active));
}

TEST_F(MapFixture, successful_owned_and_shared_map_creation)
{
  ASSERT_NO_THROW(make_owned_and_shared_row_map(mesh, active, gid_field));
}

TEST_F(MapFixture, owned_map_size)
{
  const unsigned num_elements = 8 + (bulk.parallel_size() - 1) * 4;
  ASSERT_EQ(
    make_owned_row_map(mesh, active).getGlobalNumElements(), num_elements);
}

TEST_F(MapFixture, owned_map_has_correct_local_size)
{
  // this test relies on STKs node sharing policy of lowest rank winning.  If
  // that ever changes, this will fail
  const auto asserted_local_size = (bulk.parallel_rank() == 0) ? 8u : 4u;
  ASSERT_EQ(
    make_owned_row_map(mesh, active).getLocalNumElements(),
    asserted_local_size);
}

TEST_F(MapFixture, owned_and_shared_is_just_owned_in_serial)
{
  if (bulk.parallel_size() != 1) {
    return;
  }

  const auto owned_map = make_owned_row_map(mesh, active);
  const auto oas_map = make_owned_and_shared_row_map(mesh, active, gid_field);
  ASSERT_EQ(oas_map.getGlobalNumElements(), owned_map.getGlobalNumElements());
}

TEST_F(MapFixture, owned_and_shared_globally_is_whole_shattered_mesh)
{
  const size_t ASSERTed_global_size = bulk.parallel_size() * 8;
  const auto oas_map = make_owned_and_shared_row_map(mesh, active, gid_field);
  ASSERT_EQ(oas_map.getGlobalNumElements(), ASSERTed_global_size);
}

class EntityLidFixture : public MapFixture
{
protected:
  EntityLidFixture()
    : MapFixture(),
      oas_map(make_owned_and_shared_row_map(mesh, active, gid_field))
  {
  }
  Tpetra::Map<> oas_map;
};

TEST_F(EntityLidFixture, stk_lid_to_tpetra_lid_construction)
{
  ASSERT_NO_THROW(make_stk_lid_to_tpetra_lid_map(
    mesh, active, gid_field, oas_map.getLocalMap()));
}

TEST_F(EntityLidFixture, all_active_stk_entities_have_a_valid_tpetra_lid)
{
  const auto elid = make_stk_lid_to_tpetra_lid_map(
    mesh, active, gid_field, oas_map.getLocalMap());

  const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

  auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, elid);
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto node : *ib) {
      ASSERT_NE(static_cast<unsigned>(elid(node.local_offset())), invalid);
    }
  }
}

TEST_F(EntityLidFixture, all_active_stk_entities_have_a_unique_tpetra_lid)
{
  const auto elid = make_stk_lid_to_tpetra_lid_map(
    mesh, active, gid_field, oas_map.getLocalMap());

  auto elid_h = Kokkos::create_mirror_view(elid);
  Kokkos::deep_copy(elid_h, elid);

  std::set<typename Tpetra::Map<>::local_ordinal_type> lids;
  for (const auto* ib : bulk.get_buckets(stk::topology::NODE_RANK, active)) {
    for (const auto node : *ib) {
      auto it = lids.insert(elid(node.local_offset()));
      ASSERT_TRUE(it.second);
    }
  }
}

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
