// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/NodalGradBndryElemAlg.h"

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

template <typename AlgTraits, typename PhiType, typename GradPhiType>
NodalGradBndryElemAlg<AlgTraits, PhiType, GradPhiType>::NodalGradBndryElemAlg(
  Realm& realm,
  stk::mesh::Part* part,
  PhiType* phi,
  GradPhiType* gradPhi,
  bool useShifted)
  : Algorithm(realm, part),
    dataNeeded_(realm.meta_data()),
    phi_(phi->mesh_meta_data_ordinal()),
    gradPhi_(gradPhi->mesh_meta_data_ordinal()),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    exposedAreaVec_(get_field_ordinal(
      realm_.meta_data(),
      "exposed_area_vector",
      realm_.meta_data().side_rank())),
    useShifted_(useShifted),
    meFC_(
      MasterElementRepo::get_surface_master_element_on_dev(AlgTraits::topo_))
{
  dataNeeded_.add_cvfem_face_me(meFC_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  dataNeeded_.add_coordinates_field(
    coordID, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_gathered_nodal_field(phi_, NumComp);
  dataNeeded_.add_gathered_nodal_field(dualNodalVol_, 1);
  dataNeeded_.add_face_field(
    exposedAreaVec_, AlgTraits::numFaceIp_, AlgTraits::nDim_);

  const auto shpfcn = useShifted_ ? FC_SHIFTED_SHAPE_FCN : FC_SHAPE_FCN;
  dataNeeded_.add_master_element_call(shpfcn, CURRENT_COORDINATES);
}

template <typename AlgTraits, typename PhiType, typename GradPhiType>
void
NodalGradBndryElemAlg<AlgTraits, PhiType, GradPhiType>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  using ViewHelperType = nalu_ngp::ViewHelper<ElemSimdDataType, PhiType>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto gradPhi = fieldMgr.template get_field<double>(gradPhi_);
  const auto gradPhiOps =
    nalu_ngp::simd_elem_nodal_field_updater(ngpMesh, gradPhi);

  // Bring class members into local scope for device capture
  const bool useShifted = useShifted_;
  const auto dnvID = dualNodalVol_;
  const auto exposedAreaID = exposedAreaVec_;
  const auto phiID = phi_;
  auto* meFC = meFC_;

  gradPhi.sync_to_device();
  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const std::string algName =
    (meta.get_fields()[gradPhi_]->name() + "_bndry_" +
     std::to_string(AlgTraits::topo_));
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, meta.side_rank(), dataNeeded_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* ipNodeMap = meFC->ipNodeMap();

      auto& scrView = edata.simdScrView;
      const auto& v_dnv = scrView.get_scratch_view_1D(dnvID);
      const auto& v_areav = scrView.get_scratch_view_2D(exposedAreaID);
      const ViewHelperType v_phi(scrView, phiID);

      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_shape_fcn =
        useShifted ? meViews.fc_shifted_shape_fcn : meViews.fc_shape_fcn;

      for (int di = 0; di < NumComp; ++di) {
        for (int ip = 0; ip < AlgTraits::numFaceIp_; ++ip) {
          DoubleType qIp = 0.0;
          for (int n = 0; n < AlgTraits::nodesPerFace_; ++n) {
            qIp += v_shape_fcn(ip, n) * v_phi(n, di);
          }

          const int ni = ipNodeMap[ip];
          const DoubleType inv_vol = 1.0 / v_dnv(ni);

          for (int d = 0; d < AlgTraits::nDim_; ++d) {
            DoubleType fac = qIp * v_areav(ip, d);
            gradPhiOps(edata, ni, di * AlgTraits::nDim_ + d) += fac * inv_vol;
          }
        }
      }
    });
  gradPhi.modify_on_device();
}

// NOTE: Can't use BuildTemplates here because of additional template arguments
#define INSTANTIATE_ALG(AlgTraits)                                             \
  template class NodalGradBndryElemAlg<                                        \
    AlgTraits, ScalarFieldType, VectorFieldType>;                              \
  template class NodalGradBndryElemAlg<                                        \
    AlgTraits, VectorFieldType, GenericFieldType>

INSTANTIATE_ALG(AlgTraitsTri3);
INSTANTIATE_ALG(AlgTraitsQuad4);
INSTANTIATE_ALG(AlgTraitsEdge_2D);

} // namespace nalu
} // namespace sierra
