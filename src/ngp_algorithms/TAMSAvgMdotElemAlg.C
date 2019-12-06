// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "ngp_algorithms/TAMSAvgMdotElemAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_algorithms/ViewHelper.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

template <typename AlgTraits>
TAMSAvgMdotElemAlg<AlgTraits>::TAMSAvgMdotElemAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    dataNeeded_(realm.meta_data()),
    avgTime_(get_field_ordinal(realm_.meta_data(), "rans_time_scale")),
    mdot_(get_field_ordinal(
      realm_.meta_data(), "mass_flow_rate_scs", stk::topology::ELEM_RANK)),
    avgMdot_(get_field_ordinal(
      realm_.meta_data(),
      "average_mass_flow_rate_scs",
      stk::topology::ELEM_RANK)),
    useShifted_(realm_.get_cvfem_shifted_mdot()),
    meSCS_(MasterElementRepo::get_surface_master_element<AlgTraits>())
{
  dataNeeded_.add_cvfem_surface_me(meSCS_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  dataNeeded_.add_coordinates_field(coordID, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_gathered_nodal_field(avgTime_, 1);
  dataNeeded_.add_element_field(mdot_, AlgTraits::numScsIp_);
  dataNeeded_.add_element_field(avgMdot_, AlgTraits::numScsIp_);

  dataNeeded_.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  const auto shpfcn = useShifted_ ? SCS_SHIFTED_SHAPE_FCN : SCS_SHAPE_FCN;
  dataNeeded_.add_master_element_call(shpfcn, CURRENT_COORDINATES);
}

template <typename AlgTraits>
void
TAMSAvgMdotElemAlg<AlgTraits>::execute()
{
  using ElemSimdDataType = sierra::nalu::nalu_ngp::ElemSimdData<ngp::Mesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto avgMdot = fieldMgr.template get_field<double>(avgMdot_);
  const auto avgMdotOps = nalu_ngp::simd_elem_field_updater(ngpMesh, avgMdot);

  // Bring class members into local scope for device capture
  const bool useShifted = useShifted_;
  const auto avgTimeID = avgTime_;
  const auto mdotID = mdot_;
  const auto avgMdotID = avgMdot_;
  const DoubleType dt = realm_.get_time_step();

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  nalu_ngp::run_elem_algorithm(
    "compute_avgMdot_elem_interior",
    meshInfo, stk::topology::ELEM_RANK, dataNeeded_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      auto& scrView = edata.simdScrView;
      const auto& v_avgTime = scrView.get_scratch_view_1D(avgTimeID);
      const auto& v_mdot = scrView.get_scratch_view_1D(mdotID);
      const auto& v_avgMdot = scrView.get_scratch_view_1D(avgMdotID);

      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_shape_fcn =
        useShifted ? meViews.scs_shifted_shape_fcn : meViews.scs_shape_fcn;

      for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
        DoubleType avgTimeIp = 0.0;
        for (int n = 0; n < AlgTraits::nodesPerElement_; ++n) {
          avgTimeIp += v_shape_fcn(ip, n) * v_avgTime(n);
        }

        const DoubleType weightAvg = stk::math::max(1.0 - dt / avgTimeIp, 0.0);
        const DoubleType weightInst = stk::math::min(dt / avgTimeIp, 1.0);

        avgMdotOps(edata, ip) = weightAvg * v_avgMdot(ip) + weightInst * v_mdot(ip);
      }
    });
}

INSTANTIATE_KERNEL(TAMSAvgMdotElemAlg)

} // namespace nalu
} // namespace sierra
