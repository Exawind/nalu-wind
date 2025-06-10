// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "ngp_algorithms/SDRWallFuncAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
SDRWallFuncAlg<BcAlgTraits>::SDRWallFuncAlg(
  Realm& realm, stk::mesh::Part* part, bool RANSAblBcApproach, double z0)
  : Algorithm(realm, part),
    faceData_(realm.meta_data()),
    elemData_(realm.meta_data()),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallFricVel_(get_field_ordinal(
      realm.meta_data(),
      "wall_friction_velocity_bip",
      realm.meta_data().side_rank())),
    wallArea_(get_field_ordinal(realm.meta_data(), "assembled_wall_area_sdr")),
    sdrbc_(get_field_ordinal(realm.meta_data(), "wall_model_sdr_bc")),
    sqrtBetaStar_(stk::math::sqrt(realm.get_turb_model_constant(TM_betaStar))),
    kappa_(realm.get_turb_model_constant(TM_kappa)),
    meFC_(
      MasterElementRepo::get_surface_master_element_on_dev(
        BcAlgTraits::FaceTraits::topo_)),
    meSCS_(
      MasterElementRepo::get_surface_master_element_on_dev(
        BcAlgTraits::ElemTraits::topo_)),
    RANSAblBcApproach_(RANSAblBcApproach),
    z0_(z0)
{
  faceData_.add_cvfem_face_me(meFC_);
  elemData_.add_cvfem_surface_me(meSCS_);

  faceData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(wallFricVel_, BcAlgTraits::numFaceIp_);

  elemData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
}

template <typename BcAlgTraits>
void
SDRWallFuncAlg<BcAlgTraits>::execute()
{
  using SimdDataType = nalu_ngp::FaceElemSimdData<stk::mesh::NgpMesh>;

  const auto& meta = realm_.meta_data();

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto& warea = fieldMgr.template get_field<double>(wallArea_);
  auto& sdrbc = fieldMgr.template get_field<double>(sdrbc_);
  const auto areaOps =
    nalu_ngp::simd_face_elem_nodal_field_updater(ngpMesh, warea);
  const auto sdrbcOps =
    nalu_ngp::simd_face_elem_nodal_field_updater(ngpMesh, sdrbc);

  // Bring class members into local scope for device capture
  const auto coordsID = coordinates_;
  const auto exposedAreaVecID = exposedAreaVec_;
  const auto wallFricVelID = wallFricVel_;
  const auto sqrtBetaStar = sqrtBetaStar_;
  const auto kappa = kappa_;

  auto* meFC = meFC_;
  auto* meSCS = meSCS_;
  bool RANSAblBcApproach = RANSAblBcApproach_;
  double z0 = z0_;

  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const std::string algName = "SDRWallFuncAlg_" +
                              std::to_string(BcAlgTraits::faceTopo_) + "_" +
                              std::to_string(BcAlgTraits::elemTopo_);

  nalu_ngp::run_face_elem_algorithm(
    algName, meshInfo, faceData_, elemData_, sel,
    KOKKOS_LAMBDA(SimdDataType & fdata) {
      auto& v_coord = fdata.simdElemView.get_scratch_view_2D(coordsID);
      auto& v_area = fdata.simdFaceView.get_scratch_view_2D(exposedAreaVecID);
      auto& v_fricVel = fdata.simdFaceView.get_scratch_view_1D(wallFricVelID);

      const int* faceIpNodeMap = meFC->ipNodeMap();
      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d)
          aMag += v_area(ip, d) * v_area(ip, d);
        aMag = stk::math::sqrt(aMag);

        const int nodeR = meSCS->ipNodeMap(fdata.faceOrd)[ip];
        const int nodeL = meSCS->opposingNodes(fdata.faceOrd, ip);

        DoubleType ypBip;
        if (RANSAblBcApproach) {
          // set ypBip to roughness height for wall function calculation
          ypBip = z0;
        } else {
          ypBip = 0.0;
          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            const DoubleType nj = v_area(ip, d) / aMag;
            const DoubleType ej =
              0.25 * (v_coord(nodeR, d) - v_coord(nodeL, d));
            ypBip += nj * ej * nj * ej;
          }
          ypBip = stk::math::sqrt(ypBip);
        }
        const DoubleType wallFuncSdr =
          v_fricVel(ip) / (sqrtBetaStar * kappa * ypBip);

        const int ni = faceIpNodeMap[ip];
        areaOps(fdata, ni, 0) += aMag;
        sdrbcOps(fdata, ni, 0) += wallFuncSdr * aMag;
      }
    });

  warea.modify_on_device();
  sdrbc.modify_on_device();
}

INSTANTIATE_KERNEL_FACE_ELEMENT(SDRWallFuncAlg)

} // namespace nalu
} // namespace sierra
