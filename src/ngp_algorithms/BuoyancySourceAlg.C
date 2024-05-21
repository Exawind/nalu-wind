// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/BuoyancySourceAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include "SolutionOptions.h"

namespace sierra {
namespace nalu {

BuoyancySourceAlg::BuoyancySourceAlg(
  Realm& realm, stk::mesh::Part* part, VectorFieldType* source)
  : Algorithm(realm, part),
    source_(source->mesh_meta_data_ordinal()),
    edgeAreaVec_(get_field_ordinal(
      realm_.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    coordinates_(
      get_field_ordinal(realm_.meta_data(), realm.get_coordinates_name())),
    density_(
      get_field_ordinal(realm_.meta_data(), "density", stk::mesh::StateNP1))
{
}

void
BuoyancySourceAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const int ndim = meta.spatial_dimension();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  const auto coordinates = fieldMgr.template get_field<double>(coordinates_);
  const auto density = fieldMgr.template get_field<double>(density_);
  const auto edgeAreaVec = fieldMgr.template get_field<double>(edgeAreaVec_);
  const auto dualVol = fieldMgr.template get_field<double>(dualNodalVol_);
  auto source = fieldMgr.template get_field<double>(source_);
  const auto sourceOps = nalu_ngp::edge_nodal_field_updater(ngpMesh, source);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  double gravity[3] = {0.0, 0.0, 0.0};

  if (realm_.solutionOptions_->gravity_.size() >= ndim)
    for (int idim = 0; idim < ndim; ++idim)
      gravity[idim] = realm_.solutionOptions_->gravity_[idim];

  source.sync_to_device();

  const std::string algName = meta.get_fields()[source_]->name() + "_edge";
  nalu_ngp::run_edge_algorithm(
    algName, ngpMesh, sel, KOKKOS_LAMBDA(const EntityInfoType& einfo) {
      NALU_ALIGNED DblType av[NDimMax];

      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(einfo.meshIdx, d);

      const auto nodeL = ngpMesh.fast_mesh_index(einfo.entityNodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(einfo.entityNodes[1]);

      const DblType invVolL = 1.0 / dualVol.get(nodeL, 0);
      const DblType invVolR = 1.0 / dualVol.get(nodeR, 0);

      const DblType rhoIp =
        0.5 * (density.get(nodeL, 0) + density.get(nodeR, 0));

      DblType cc_face[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < ndim; ++i)
        cc_face[i] =
          0.5 * (coordinates.get(nodeL, i) + coordinates.get(nodeR, i));

      DblType gravity_left = 0.0;
      DblType gravity_right = 0.0;

      // There is a better form of this that should work better for general
      // meshes. It will involve calculating an effective length to conserve
      // volume. This one was used by Denner & van Wechem, 2014 successfully and
      // is simpler.
      for (int i = 0; i < ndim; ++i) {
        gravity_left += gravity[i] * (cc_face[i] - coordinates.get(nodeL, i));
        gravity_right += gravity[i] * (cc_face[i] - coordinates.get(nodeR, i));
      }

      for (int i = 0; i < 3; ++i) {
        sourceOps(einfo, 0, i) += av[i] * rhoIp * gravity_left * invVolL;
        sourceOps(einfo, 1, i) -= av[i] * rhoIp * gravity_right * invVolR;
      }
    });
  source.modify_on_device();
}

} // namespace nalu
} // namespace sierra
