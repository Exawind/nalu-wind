// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/GeometryInteriorAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
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
GeometryInteriorAlg<AlgTraits>::GeometryInteriorAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    dataNeeded_(realm.meta_data()),
    dualNodalVol_(get_field_ordinal(realm_.meta_data(), "dual_nodal_volume")),
    elemVol_(get_field_ordinal(
      realm_.meta_data(), "element_volume", stk::topology::ELEM_RANK)),
    meSCV_(
      MasterElementRepo::get_volume_master_element_on_dev(AlgTraits::topo_)),
    meSCS_(
      MasterElementRepo::get_surface_master_element_on_dev(AlgTraits::topo_))
{
  dataNeeded_.add_cvfem_volume_me(meSCV_);
  dataNeeded_.add_cvfem_surface_me(meSCS_);

  const auto coordID = get_field_ordinal(
    realm_.meta_data(), realm_.solutionOptions_->get_coordinates_name());
  dataNeeded_.add_coordinates_field(
    coordID, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataNeeded_.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);

  if (realm_.realmUsesEdges_) {
    edgeAreaVec_ = get_field_ordinal(
      realm_.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK);
    dataNeeded_.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  }
}

template <typename AlgTraits>
void
GeometryInteriorAlg<AlgTraits>::execute()
{
  if (realm_.checkJacobians_) {
    try {
      impl_negative_jacobian_check();
    } catch (const std::exception& e) {
      // dump exodus file if the user enabled this feature then rethrow
      realm_.provide_output(realm_.outputFailedJacobians_);
      throw e;
    }
  }

  impl_compute_dual_nodal_volume();

  if (realm_.realmUsesEdges_)
    impl_compute_edge_area_vector();
}

template <typename AlgTraits>
void
GeometryInteriorAlg<AlgTraits>::impl_compute_dual_nodal_volume()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto dualVol = fieldMgr.template get_field<double>(dualNodalVol_);
  auto elemVol = fieldMgr.template get_field<double>(elemVol_);
  const auto dnvOps = nalu_ngp::simd_elem_nodal_field_updater(ngpMesh, dualVol);
  const auto elemVolOps = nalu_ngp::simd_elem_field_updater(ngpMesh, elemVol);
  MasterElement* meSCV = meSCV_;
  dualVol.sync_to_device();
  elemVol.sync_to_device();

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const std::string algName = "compute_dnv_" + std::to_string(AlgTraits::topo_);
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, stk::topology::ELEM_RANK, dataNeeded_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* ipNodeMap = meSCV->ipNodeMap();
      auto& scrView = edata.simdScrView;
      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_scv_vol = meViews.scv_volume;

      elemVolOps(edata, 0) = 0.0;
      for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
        const auto nn = ipNodeMap[ip];
        dnvOps(edata, nn, 0) += v_scv_vol(ip);
        elemVolOps(edata, 0) += v_scv_vol(ip);
      }
    });

  dualVol.modify_on_device();
  elemVol.modify_on_device();
}

template <typename AlgTraits>
void
GeometryInteriorAlg<AlgTraits>::impl_negative_jacobian_check()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  size_t numNegVol = 0;
  Kokkos::Sum<size_t> reducer(numNegVol);
  const std::string algName =
    "negative_volume_check_" + std::to_string(AlgTraits::topo_);
  nalu_ngp::run_elem_par_reduce(
    algName, meshInfo, stk::topology::ELEM_RANK, dataNeeded_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata, size_t & threadVal) {
      auto& scrView = edata.simdScrView;
      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_scv_vol = meViews.scv_volume;

      for (int ip = 0; ip < AlgTraits::numScvIp_; ++ip) {
        for (int i = 0; i < edata.numSimdElems; ++i) {
          if (v_scv_vol(ip)[i] < 0.0)
            ++threadVal;
        }
      }
    },
    reducer);

  if (numNegVol > 0) {
    const stk::topology topology(AlgTraits::topo_);
    throw std::runtime_error(
      "GeometryInteriorAlg encountered " + std::to_string(numNegVol) +
      " negative sub-control volumes for topology " +
      std::to_string(AlgTraits::topo_) + "  name: " + topology.char_name());
  }
}

template <typename AlgTraits>
void
GeometryInteriorAlg<AlgTraits>::impl_compute_edge_area_vector()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto edgeAreaVec = fieldMgr.template get_field<double>(edgeAreaVec_);
  MasterElement* meSCS = meSCS_;

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const std::string algName =
    "compute_edge_areav_" + std::to_string(AlgTraits::topo_);
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, stk::topology::ELEM_RANK, dataNeeded_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* lrscv = meSCS->adjacentNodes();
      const int* scsIpEdgeMap = meSCS->scsIpEdgeOrd();

      auto& scrView = edata.simdScrView;
      const auto& meViews = scrView.get_me_views(CURRENT_COORDINATES);
      const auto& v_areav = meViews.scs_areav;

      // Manually de-interleave here
      for (int si = 0; si < edata.numSimdElems; ++si) {
        const auto edges = ngpMesh.get_edges(
          stk::topology::ELEM_RANK,
          ngpMesh.fast_mesh_index(edata.elemInfo[si].entity));
        for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {
          // Edge for this integration point
          const int nedge = scsIpEdgeMap[ip];
          // Index of "left" node in the element relations
          const int iLn = lrscv[2 * ip];

          // Nodes connected to this edge
          const auto edgeID = ngpMesh.fast_mesh_index(edges[nedge]);
          const auto edge_nodes =
            ngpMesh.get_nodes(stk::topology::EDGE_RANK, edgeID);

          // Left node comparison
          const auto lnElemId = edata.elemInfo[si].entityNodes[iLn];
          const auto lnEdgeId = edge_nodes[0];

          const double sign = (lnElemId == lnEdgeId) ? 1.0 : -1.0;

          for (int d = 0; d < AlgTraits::nDim_; ++d) {
            Kokkos::atomic_add(
              &edgeAreaVec.get(edgeID, d),
              stk::simd::get_data(v_areav(ip, d), si) * sign);
          }
        }
      }
    });

  edgeAreaVec.modify_on_device();
}

INSTANTIATE_KERNEL(GeometryInteriorAlg)

} // namespace nalu
} // namespace sierra
