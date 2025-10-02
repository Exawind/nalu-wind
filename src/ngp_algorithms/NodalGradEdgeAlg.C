// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/NodalGradEdgeAlg.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename PhiType, typename GradPhiType>
NodalGradEdgeAlg<PhiType, GradPhiType>::NodalGradEdgeAlg(
  Realm& realm, stk::mesh::Part* part, PhiType* phi, GradPhiType* gradPhi)
  : Algorithm(realm, part),
    phi_(phi->mesh_meta_data_ordinal()),
    gradPhi_(gradPhi->mesh_meta_data_ordinal()),
    edgeAreaVec_(get_field_ordinal(
      realm_.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    dim1_(max_extent(*phi, 0)),
    dim2_(realm_.meta_data().spatial_dimension())
{
  const int gradPhiSize = max_extent(*gradPhi, 0);
  if (dim1_ == 1) {
    STK_ThrowRequireMsg(
      gradPhiSize == dim2_, "NodalGradEdgeAlg called with scalar input field '"
                              << phi->name()
                              << "' but with non-vector output field '"
                              << gradPhi->name() << "' of length "
                              << gradPhiSize << " (should be " << dim2_ << ")");
  } else if (dim1_ == dim2_) {
    STK_ThrowRequireMsg(
      gradPhiSize == dim2_ * dim2_,
      "NodalGradBndryElemAlg called with vector input field '"
        << phi->name() << "' but with non-tensor output field '"
        << gradPhi->name() << "' of length " << gradPhiSize << " (should be "
        << dim2_ * dim2_ << ")");
  } else {
    STK_ThrowErrorMsg(
      "NodalGradBndryElemAlg called with an input field '"
      << phi->name()
      << "' that is not a scalar or a vector.  "
         "Actual length = "
      << dim1_);
  }
}

template <typename PhiType, typename GradPhiType>
void
NodalGradEdgeAlg<PhiType, GradPhiType>::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  const auto phi = fieldMgr.template get_field<double>(phi_);
  const auto edgeAreaVec = fieldMgr.template get_field<double>(edgeAreaVec_);
  const auto dualVol = fieldMgr.template get_field<double>(dualNodalVol_);
  auto gradPhi = fieldMgr.template get_field<double>(gradPhi_);
  const auto gradPhiOps = nalu_ngp::edge_nodal_field_updater(ngpMesh, gradPhi);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  // Bring class members into local scope for device capture
  const int dim1 = dim1_;
  const int dim2 = dim2_;

  gradPhi.sync_to_device();

  const std::string algName = meta.get_fields()[gradPhi_]->name() + "_edge";
  nalu_ngp::run_edge_algorithm(
    algName, ngpMesh, sel, KOKKOS_LAMBDA(const EntityInfoType& einfo) {
       DblType av[NDimMax];

      for (int d = 0; d < dim2; ++d)
        av[d] = edgeAreaVec.get(einfo.meshIdx, d);

      const auto nodeL = ngpMesh.fast_mesh_index(einfo.entityNodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(einfo.entityNodes[1]);

      const DblType invVolL = 1.0 / dualVol.get(nodeL, 0);
      const DblType invVolR = 1.0 / dualVol.get(nodeR, 0);

      int counter = 0;
      for (int i = 0; i < dim1; ++i) {
        const double phiIp = 0.5 * (phi.get(nodeL, i) + phi.get(nodeR, i));

        for (int j = 0; j < dim2; ++j) {
          const DblType ajPhiIp = av[j] * phiIp;
          gradPhiOps(einfo, 0, counter) += ajPhiIp * invVolL;
          gradPhiOps(einfo, 1, counter) -= ajPhiIp * invVolR;
          counter++;
        }
      }
    });
  gradPhi.modify_on_device();
}

template class NodalGradEdgeAlg<ScalarFieldType, VectorFieldType>;

} // namespace nalu
} // namespace sierra
