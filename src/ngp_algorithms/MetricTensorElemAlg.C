// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/MetricTensorElemAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_algorithms/ViewHelper.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MetricTensorElemAlg<AlgTraits>::MetricTensorElemAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    dataNeeded_(realm.meta_data()),
    nodalMij_(get_field_ordinal(realm.meta_data(), "metric_tensor")),
    dualNodalVol_(get_field_ordinal(realm.meta_data(), "dual_nodal_volume")),
    meSCV_(MasterElementRepo::get_volume_master_element<AlgTraits>())
{
  dataNeeded_.add_cvfem_volume_me(meSCV_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  dataNeeded_.add_coordinates_field(
    coordID, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_gathered_nodal_field(
    nodalMij_, AlgTraits::nDim_ * AlgTraits::nDim_);
  dataNeeded_.add_gathered_nodal_field(dualNodalVol_, 1);

  dataNeeded_.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataNeeded_.add_master_element_call(SCV_GRAD_OP, CURRENT_COORDINATES);
  dataNeeded_.add_master_element_call(SCV_MIJ, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
MetricTensorElemAlg<AlgTraits>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto Mij = fieldMgr.template get_field<double>(nodalMij_);
  const auto MijOps = nalu_ngp::simd_elem_nodal_field_updater(ngpMesh, Mij);

  // Bring class members into local scope for device capture
  const auto dnvID = dualNodalVol_;
  auto* meSCV = meSCV_;

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  nalu_ngp::run_elem_algorithm(
    "computeMetricTensorAlg", meshInfo, stk::topology::ELEM_RANK, dataNeeded_,
    sel, KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      auto& scrView = edata.simdScrView;
      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_scv_volume = meViews.scv_volume;
      const auto& v_scv_mij = meViews.metric;
      const auto* ipNodeMap = meSCV->ipNodeMap();
      const auto& v_dnv = scrView.get_scratch_view_1D(dnvID);

      for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
        const int nearestNode = ipNodeMap[ip];

        for (int i = 0; i < AlgTraits::nDim_; ++i)
          for (int j = 0; j < AlgTraits::nDim_; ++j) {
            MijOps(edata, nearestNode, i * AlgTraits::nDim_ + j) +=
              v_scv_mij(ip, i, j) * v_scv_volume(ip) / v_dnv(ip);
          }
      }
    });
}

INSTANTIATE_KERNEL(MetricTensorElemAlg)

} // namespace nalu
} // namespace sierra
