// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/NodalGradElemAlg.h"

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
NodalGradElemAlg<AlgTraits, PhiType, GradPhiType>::NodalGradElemAlg(
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
    useShifted_(useShifted),
    meSCS_(
      MasterElementRepo::get_surface_master_element_on_dev(AlgTraits::topo_))
{
  dataNeeded_.add_cvfem_surface_me(meSCS_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  dataNeeded_.add_coordinates_field(
    coordID, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_gathered_nodal_field(phi_, NumComp);
  dataNeeded_.add_gathered_nodal_field(dualNodalVol_, 1);

  dataNeeded_.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  const auto shpfcn = useShifted_ ? SCS_SHIFTED_SHAPE_FCN : SCS_SHAPE_FCN;
  dataNeeded_.add_master_element_call(shpfcn, CURRENT_COORDINATES);
}

template <typename AlgTraits, typename PhiType, typename GradPhiType>
void
NodalGradElemAlg<AlgTraits, PhiType, GradPhiType>::execute()
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
  const auto phiID = phi_;
  auto* meSCS = meSCS_;

  gradPhi.sync_to_device();

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const std::string algName =
    (meta.get_fields()[gradPhi_]->name() + "_elem_" +
     std::to_string(AlgTraits::topo_));
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, stk::topology::ELEM_RANK, dataNeeded_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* lrscv = meSCS->adjacentNodes();

      auto& scrView = edata.simdScrView;
      const auto& v_dnv = scrView.get_scratch_view_1D(dnvID);
      const ViewHelperType v_phi(scrView, phiID);

      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_areav = meViews.scs_areav;
      const auto& v_shape_fcn =
        useShifted ? meViews.scs_shifted_shape_fcn : meViews.scs_shape_fcn;

      for (int di = 0; di < NumComp; ++di) {
        for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
          DoubleType qIp = 0.0;
          for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
            qIp += v_shape_fcn(ip, n) * v_phi(n, di);
          }

          int il = lrscv[2 * ip];
          int ir = lrscv[2 * ip + 1];

          for (int d = 0; d < AlgTraits::nDim_; ++d) {
            DoubleType fac = qIp * v_areav(ip, d);
            DoubleType valL = fac / v_dnv(il);
            DoubleType valR = fac / v_dnv(ir);

            gradPhiOps(edata, il, di * AlgTraits::nDim_ + d) += valL;
            gradPhiOps(edata, ir, di * AlgTraits::nDim_ + d) -= valR;
          }
        }
      }
    });

  gradPhi.modify_on_device();
}

// NOTE: Can't use BuildTemplates here because of additional template arguments
#define INSTANTIATE_ALG(AlgTraits)                                             \
  template class NodalGradElemAlg<                                             \
    AlgTraits, ScalarFieldType, VectorFieldType>;                              \
  template class NodalGradElemAlg<AlgTraits, VectorFieldType, GenericFieldType>

INSTANTIATE_ALG(AlgTraitsHex8);
INSTANTIATE_ALG(AlgTraitsTet4);
INSTANTIATE_ALG(AlgTraitsPyr5);
INSTANTIATE_ALG(AlgTraitsWed6);
INSTANTIATE_ALG(AlgTraitsTri3_2D);
INSTANTIATE_ALG(AlgTraitsQuad4_2D);

} // namespace nalu
} // namespace sierra
