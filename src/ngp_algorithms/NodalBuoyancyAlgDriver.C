// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/NodalBuoyancyAlgDriver.h"
#include "ngp_utils/NgpFieldUtils.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "Realm.h"

#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_mesh/base/FieldBLAS.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/NgpFieldParallel.hpp"

namespace sierra {
namespace nalu {

NodalBuoyancyAlgDriver::NodalBuoyancyAlgDriver(
  Realm& realm,
  const std::string& sourceName,
  const std::string& sourceweightName)
  : NgpAlgDriver(realm),
    sourceName_(sourceName),
    sourceweightName_(sourceweightName)
{
}

void
NodalBuoyancyAlgDriver::pre_work()
{
  const auto& meta = realm_.meta_data();

  auto* source =
    meta.template get_field<double>(stk::topology::NODE_RANK, sourceName_);

  auto* sourceweight = meta.template get_field<double>(
    stk::topology::NODE_RANK, sourceweightName_);

  stk::mesh::field_fill(0.0, *source);
  stk::mesh::field_fill(0.0, *sourceweight);

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto ngpsource =
    fieldMgr.template get_field<double>(source->mesh_meta_data_ordinal());
  auto ngpsourceweight =
    fieldMgr.template get_field<double>(sourceweight->mesh_meta_data_ordinal());

  ngpsource.set_all(ngpMesh, 0.0);
  ngpsource.clear_sync_state();

  ngpsourceweight.set_all(ngpMesh, 0.0);
  ngpsourceweight.clear_sync_state();
}

void
NodalBuoyancyAlgDriver::post_work()
{
  // TODO: Revisit logic after STK updates to ngp parallel updates
  const auto& meta = realm_.meta_data();
  const auto& bulk = realm_.bulk_data();
  const auto& meshInfo = realm_.mesh_info();

  auto* sourceweight = meta.template get_field<double>(
    stk::topology::NODE_RANK, sourceweightName_);
  auto& ngpsourceweight = nalu_ngp::get_ngp_field(meshInfo, sourceweightName_);

  auto* source =
    meta.template get_field<double>(stk::topology::NODE_RANK, sourceName_);
  auto& ngpsource = nalu_ngp::get_ngp_field(meshInfo, sourceName_);

  comm::scatter_sum(bulk, {sourceweight, source});

  // Divide by weight here

  using Traits = nalu_ngp::NGPMeshTraits<>;

  const auto& ngpMesh = meshInfo.ngp_mesh();

  stk::mesh::Selector sel =
    (meta.locally_owned_part() | meta.globally_shared_part()) &
    stk::mesh::selectField(*source);

  const int dim2 = meta.spatial_dimension();

  nalu_ngp::run_entity_algorithm(
    "apply_weight_to_buoyancy", ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const Traits::MeshIndex& mi) {
      if (ngpsourceweight.get(mi, 0) > 1e-12) {
        for (int idim = 0; idim < dim2; ++idim) {
          ngpsource.get(mi, idim) =
            ngpsource.get(mi, idim) / ngpsourceweight.get(mi, 0);
        }
      }
    });

  ngpsource.modify_on_device();
}

} // namespace nalu
} // namespace sierra
