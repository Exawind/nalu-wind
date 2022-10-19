// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gcl/MeshVelocityEdgeAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "master_element/Hex8GeometryFunctions.h"
#include "ngp_algorithms/ViewHelper.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"

#include <cmath>

namespace sierra {
namespace nalu {

template <typename AlgTraits>
MeshVelocityEdgeAlg<AlgTraits>::MeshVelocityEdgeAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    elemData_(realm.meta_data()),
    modelCoords_(get_field_ordinal(realm.meta_data(), "coordinates")),
    currentCoords_(get_field_ordinal(realm.meta_data(), "current_coordinates")),
    meshDispNp1_(get_field_ordinal(
      realm.meta_data(), "mesh_displacement", stk::mesh::StateNP1)),
    meshDispN_(get_field_ordinal(
      realm.meta_data(), "mesh_displacement", stk::mesh::StateN)),
    edgeFaceVelMag_(get_field_ordinal(
      realm.meta_data(), "edge_face_velocity_mag", stk::topology::EDGE_RANK)),
    edgeSweptVolumeNp1_(get_field_ordinal(
      realm.meta_data(),
      "edge_swept_face_volume",
      stk::mesh::StateNP1,
      stk::topology::EDGE_RANK)),
    edgeSweptVolumeN_(get_field_ordinal(
      realm.meta_data(),
      "edge_swept_face_volume",
      stk::mesh::StateN,
      stk::topology::EDGE_RANK)),
    meSCS_(MasterElementRepo::get_surface_master_element<AlgTraits>()),
    isoCoordsShapeFcn_("isoCoordShapFcn", 152)
{

  elemData_.add_cvfem_surface_me(meSCS_);

  elemData_.add_coordinates_field(
    modelCoords_, AlgTraits::nDim_, MODEL_COORDINATES);
  elemData_.add_coordinates_field(
    currentCoords_, AlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_gathered_nodal_field(meshDispNp1_, AlgTraits::nDim_);
  elemData_.add_gathered_nodal_field(meshDispN_, AlgTraits::nDim_);

  elemData_.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);

  Kokkos::View<double*, sierra::nalu::MemSpace> isoShapeHost(
    "isoShapHost", 152);
  meSCS_->general_shape_fcn(19, isoParCoords_, &isoShapeHost(0));
  Kokkos::deep_copy(isoCoordsShapeFcn_, isoShapeHost);

  if (!std::is_same<AlgTraits, AlgTraitsHex8>::value) {
    throw std::runtime_error("MeshVelocityEdgeAlg is only supported for Hex8");
  }
}

template <typename AlgTraits>
void
MeshVelocityEdgeAlg<AlgTraits>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;

  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const DoubleType dt = realm_.get_time_step();
  const DoubleType gamma1 = realm_.get_gamma1();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto edgeFaceVelMag = fieldMgr.template get_field<double>(edgeFaceVelMag_);
  auto edgeSweptVol = fieldMgr.template get_field<double>(edgeSweptVolumeNp1_);

  const auto modelCoordsID = modelCoords_;
  const auto meshDispNp1ID = meshDispNp1_;
  const auto meshDispNID = meshDispN_;
  MasterElement* meSCS = meSCS_;
  const auto isoCoordsShapeFcn = isoCoordsShapeFcn_;
  const auto scsFaceNodeMap = scsFaceNodeMap_;

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  const std::string algName =
    "compute_mesh_vel_" + std::to_string(AlgTraits::topo_);
  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, stk::topology::ELEM_RANK, elemData_, sel,
    KOKKOS_LAMBDA(ElemSimdDataType & edata) {
      const int* lrscv = meSCS->adjacentNodes();
      const int* scsIpEdgeMap = meSCS->scsIpEdgeOrd();

      auto& scrView = edata.simdScrView;

      const auto& mCoords = scrView.get_scratch_view_2D(modelCoordsID);
      const auto& dispNp1 = scrView.get_scratch_view_2D(meshDispNp1ID);
      const auto& dispN = scrView.get_scratch_view_2D(meshDispNID);

      DoubleType scs_coords_n[19][AlgTraits::nDim_];
      DoubleType scs_coords_np1[19][AlgTraits::nDim_];

      for (int i = 0; i < 19; i++) {
        for (int j = 0; j < AlgTraits::nDim_; j++) {
          scs_coords_n[i][j] = 0.0;
          scs_coords_np1[i][j] = 0.0;
        }
        for (int k = 0; k < AlgTraits::nodesPerElement_; k++) {
          const DoubleType r =
            isoCoordsShapeFcn(i * AlgTraits::nodesPerElement_ + k);
          for (int j = 0; j < AlgTraits::nDim_; j++) {
            scs_coords_n[i][j] += r * (mCoords(k, j) + dispN(k, j));
            scs_coords_np1[i][j] += r * (mCoords(k, j) + dispNp1(k, j));
          }
        }
      }

      for (int ip = 0; ip < AlgTraits::numScsIp_; ++ip) {

        const int na = scsFaceNodeMap[ip][0];
        const int nb = scsFaceNodeMap[ip][1];
        const int nc = scsFaceNodeMap[ip][2];
        const int nd = scsFaceNodeMap[ip][3];

        DoubleType scs_vol_coords[8][3];

        for (int j = 0; j < AlgTraits::nDim_; j++) {
          scs_vol_coords[0][j] = scs_coords_n[na][j];
          scs_vol_coords[1][j] = scs_coords_n[nb][j];
          scs_vol_coords[2][j] = scs_coords_n[nc][j];
          scs_vol_coords[3][j] = scs_coords_n[nd][j];
          scs_vol_coords[4][j] = scs_coords_np1[na][j];
          scs_vol_coords[5][j] = scs_coords_np1[nb][j];
          scs_vol_coords[6][j] = scs_coords_np1[nc][j];
          scs_vol_coords[7][j] = scs_coords_np1[nd][j];
        }

        DoubleType tmp = hex_volume_grandy(scs_vol_coords);
        DoubleType tmp2 = gamma1 * tmp / dt;

        for (int si = 0; si < edata.numSimdElems; ++si) {
          const auto edges = ngpMesh.get_edges(
            stk::topology::ELEM_RANK,
            ngpMesh.fast_mesh_index(edata.elemInfo[si].entity));

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

          Kokkos::atomic_add(
            &edgeSweptVol.get(edgeID, 0), stk::simd::get_data(tmp, si) * sign);
          Kokkos::atomic_add(
            &edgeFaceVelMag.get(edgeID, 0),
            stk::simd::get_data(tmp2, si) * sign);
        }
      }
    });
}

INSTANTIATE_KERNEL(MeshVelocityEdgeAlg)

} // namespace nalu
} // namespace sierra
