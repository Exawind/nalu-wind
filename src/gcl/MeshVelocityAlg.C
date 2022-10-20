// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "gcl/MeshVelocityAlg.h"
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
#include <type_traits>
#include <cassert>

namespace sierra {
namespace nalu {

// fixed for hex right now
constexpr int NUM_IP = 19;

template <typename AlgTraits>
MeshVelocityAlg<AlgTraits>::MeshVelocityAlg(Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    elemData_(realm.meta_data()),
    modelCoords_(get_field_ordinal(realm.meta_data(), "coordinates")),
    currentCoords_(get_field_ordinal(realm.meta_data(), "current_coordinates")),
    meshDispNp1_(get_field_ordinal(
      realm.meta_data(), "mesh_displacement", stk::mesh::StateNP1)),
    meshDispN_(get_field_ordinal(
      realm.meta_data(), "mesh_displacement", stk::mesh::StateN)),
    faceVelMag_(get_field_ordinal(
      realm.meta_data(), "face_velocity_mag", stk::topology::ELEM_RANK)),
    sweptVolumeNp1_(get_field_ordinal(
      realm.meta_data(),
      "swept_face_volume",
      stk::mesh::StateNP1,
      stk::topology::ELEM_RANK)),
    sweptVolumeN_(get_field_ordinal(
      realm.meta_data(),
      "swept_face_volume",
      stk::mesh::StateN,
      stk::topology::ELEM_RANK)),
    meSCS_(MasterElementRepo::get_surface_master_element<AlgTraits>()),
    isoCoordsShapeFcn_("isoCoordShapFcn", 152)
{
  if (!std::is_same<AlgTraits, AlgTraitsHex8>::value) {
    throw std::runtime_error("MeshVelocityEdgeAlg is only supported for Hex8");
  }

  elemData_.add_cvfem_surface_me(meSCS_);

  elemData_.add_coordinates_field(
    modelCoords_, AlgTraits::nDim_, MODEL_COORDINATES);
  elemData_.add_coordinates_field(
    currentCoords_, AlgTraits::nDim_, CURRENT_COORDINATES);
  elemData_.add_element_field(faceVelMag_, AlgTraits::numScsIp_);
  elemData_.add_element_field(sweptVolumeN_, AlgTraits::numScsIp_);
  elemData_.add_element_field(sweptVolumeNp1_, AlgTraits::numScsIp_);
  elemData_.add_gathered_nodal_field(meshDispNp1_, AlgTraits::nDim_);
  elemData_.add_gathered_nodal_field(meshDispN_, AlgTraits::nDim_);

  elemData_.add_master_element_call(SCS_AREAV, CURRENT_COORDINATES);
  Kokkos::View<double*, sierra::nalu::MemSpace> isoShapeHost(
    "isoShapHost", 152);
  meSCS_->general_shape_fcn(NUM_IP, isoParCoords_, &isoShapeHost(0));
  Kokkos::deep_copy(isoCoordsShapeFcn_, isoShapeHost);

} // namespace nalu

template <typename AlgTraits>
void
MeshVelocityAlg<AlgTraits>::execute()
{
  using ElemSimdDataType =
    sierra::nalu::nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  const auto& meshInfo = realm_.mesh_info();
  const auto& meta = meshInfo.meta();
  const DoubleType dt = realm_.get_time_step();
  const DoubleType gamma1 = realm_.get_gamma1();
  const DoubleType gamma2 = realm_.get_gamma2();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto faceVel = fieldMgr.template get_field<double>(faceVelMag_);
  auto ngpSweptVol = fieldMgr.template get_field<double>(sweptVolumeNp1_);
  const auto faceVelOps = nalu_ngp::simd_elem_field_updater(ngpMesh, faceVel);
  const auto sweptVolOps =
    nalu_ngp::simd_elem_field_updater(ngpMesh, ngpSweptVol);

  const auto modelCoordsID = modelCoords_;
  const auto meshDispNp1ID = meshDispNp1_;
  const auto meshDispNID = meshDispN_;
  const auto sweptVolNID = sweptVolumeN_;
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
      auto& scrView = edata.simdScrView;

      const auto& mCoords = scrView.get_scratch_view_2D(modelCoordsID);
      const auto& dispNp1 = scrView.get_scratch_view_2D(meshDispNp1ID);
      const auto& dispN = scrView.get_scratch_view_2D(meshDispNID);
      const auto& sweptVolN = scrView.get_scratch_view_1D(sweptVolNID);

      DoubleType dx[NUM_IP][AlgTraits::nDim_];
      DoubleType scs_coords_n[NUM_IP][AlgTraits::nDim_];
      DoubleType scs_coords_np1[NUM_IP][AlgTraits::nDim_];

      for (int i = 0; i < NUM_IP; i++) {
        for (int j = 0; j < AlgTraits::nDim_; j++) {
          dx[i][j] = 0.0;
          scs_coords_n[i][j] = 0.0;
          scs_coords_np1[i][j] = 0.0;
        }
        for (int k = 0; k < AlgTraits::nodesPerElement_; k++) {
          const DoubleType r =
            isoCoordsShapeFcn(i * AlgTraits::nodesPerElement_ + k);
          for (int j = 0; j < AlgTraits::nDim_; j++) {
            dx[i][j] += r * (dispNp1(k, j) - dispN(k, j));
            scs_coords_n[i][j] += r * (mCoords(k, j) + dispN(k, j));
            scs_coords_np1[i][j] += r * (mCoords(k, j) + dispNp1(k, j));
          }
        }
      }

      constexpr int Hex8numScsIp = AlgTraitsHex8::numScsIp_;
      const int nip = std::min(Hex8numScsIp, AlgTraits::numScsIp_);

      for (int ip = 0; ip < nip; ++ip) {

        DoubleType scs_vol_coords[8][3];

        for (int j = 0; j < AlgTraits::nDim_; j++) {
          scs_vol_coords[0][j] = scs_coords_n[scsFaceNodeMap[ip][0]][j];
          scs_vol_coords[1][j] = scs_coords_n[scsFaceNodeMap[ip][1]][j];
          scs_vol_coords[2][j] = scs_coords_n[scsFaceNodeMap[ip][2]][j];
          scs_vol_coords[3][j] = scs_coords_n[scsFaceNodeMap[ip][3]][j];
          scs_vol_coords[4][j] = scs_coords_np1[scsFaceNodeMap[ip][0]][j];
          scs_vol_coords[5][j] = scs_coords_np1[scsFaceNodeMap[ip][1]][j];
          scs_vol_coords[6][j] = scs_coords_np1[scsFaceNodeMap[ip][2]][j];
          scs_vol_coords[7][j] = scs_coords_np1[scsFaceNodeMap[ip][3]][j];
        }
        DoubleType tmp = hex_volume_grandy(scs_vol_coords);

        sweptVolOps(edata, ip) = tmp;

        faceVelOps(edata, ip) =
          (gamma1 * tmp + (gamma1 + gamma2) * sweptVolN(ip)) / dt;
      }
    });
}

INSTANTIATE_KERNEL(MeshVelocityAlg)

} // namespace nalu
} // namespace sierra
