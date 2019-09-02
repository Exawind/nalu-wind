/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

template <typename PhiType, typename GradPhiType>
NodalGradEdgeAlg<PhiType, GradPhiType>::NodalGradEdgeAlg(
  Realm& realm,
  stk::mesh::Part* part,
  PhiType* phi,
  GradPhiType* gradPhi)
  : Algorithm(realm, part),
    phi_(phi->mesh_meta_data_ordinal()),
    gradPhi_(gradPhi->mesh_meta_data_ordinal()),
    edgeAreaVec_(get_field_ordinal(
      realm_.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    dim1(std::is_same<PhiType, ScalarFieldType>::value ? 1 : realm_.spatialDimension_),
    dim2(realm_.spatialDimension_)
{}

template <typename PhiType, typename GradPhiType>
void NodalGradEdgeAlg<PhiType, GradPhiType>::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<ngp::Mesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const auto phi = fieldMgr.template get_field<double>(phi_);
  const auto edgeAreaVec = fieldMgr.template get_field<double>(edgeAreaVec_);
  const auto dualVol = fieldMgr.template get_field<double>(dualNodalVol_);
  auto gradPhi = fieldMgr.template get_field<double>(gradPhi_);

  const stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(partVec_)
    & !(realm_.get_inactive_selector());

  nalu_ngp::run_edge_algorithm(
    ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& einfo) {
      const auto gradPhiOps = nalu_ngp::edge_nodal_field_updater(
        ngpMesh, gradPhi, einfo);
      NALU_ALIGNED DblType av[NDimMax];

      for (int d=0; d < dim2; ++d)
        av[d] = edgeAreaVec.get(einfo.meshIdx, d);

      const auto& nodes = einfo.entityNodes;
      const auto nodeL = nodes[0];
      const auto nodeR = nodes[1];

      const DblType invVolL = 1.0 / dualVol.get(ngpMesh, nodeL, 0);
      const DblType invVolR = 1.0 / dualVol.get(ngpMesh, nodeR, 0);

      int counter = 0;
      for (int i = 0; i < dim1; ++i) {
        const double phiIp = 0.5 * (
          phi.get(ngpMesh, nodeL, i) + phi.get(ngpMesh, nodeR, i));

        for (int j=0; j < dim2; ++j) {
          const DblType ajPhiIp = av[j] * phiIp;
          gradPhiOps(0, counter) += ajPhiIp * invVolL;
          gradPhiOps(1, counter) -= ajPhiIp * invVolR;
          counter++;
        }
      }
    });
}

template class NodalGradEdgeAlg<ScalarFieldType, VectorFieldType>;
template class NodalGradEdgeAlg<VectorFieldType, GenericFieldType>;

}  // nalu
}  // sierra
