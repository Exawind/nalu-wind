/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "ngp_algorithms/WallFuncGeometryAlg.h"

#include "BuildTemplates.h"
#include "master_element/MasterElementFactory.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldOps.h"
#include "Realm.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
WallFuncGeometryAlg<BcAlgTraits>::WallFuncGeometryAlg(
  Realm& realm,
  stk::mesh::Part* part)
  : Algorithm(realm, part),
    faceData_(realm.meta_data()),
    elemData_(realm.meta_data()),
    coordinates_(
      get_field_ordinal(realm.meta_data(), realm.get_coordinates_name())),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    wallNormDistBip_(get_field_ordinal(
                   realm.meta_data(),
                   "wall_normal_distance_bip",
                   realm.meta_data().side_rank())),
    wallArea_(get_field_ordinal(realm.meta_data(), "assembled_wall_area_wf")),
    wallNormDist_(get_field_ordinal(realm.meta_data(), "assembled_wall_normal_distance")),
    meFC_(MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  faceData_.add_cvfem_face_me(meFC_);
  elemData_.add_cvfem_surface_me(meSCS_);

  faceData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);

  elemData_.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
}

template<typename BcAlgTraits>
void WallFuncGeometryAlg<BcAlgTraits>::execute()
{
  using MeshIndex = nalu_ngp::NGPMeshTraits<ngp::Mesh>::MeshIndex;
  using SimdDataType = nalu_ngp::FaceElemSimdData<ngp::Mesh>;

  const auto& meta = realm_.meta_data();

  const auto& meshInfo = realm_.mesh_info();
  const auto ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();
  auto wdistBip = fieldMgr.template get_field<double>(wallNormDistBip_);
  auto wdist = fieldMgr.template get_field<double>(wallNormDist_);
  auto warea = fieldMgr.template get_field<double>(wallArea_);

  const stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(partVec_);

  // Bring class members into local scope for device capture
  const unsigned coordsID = coordinates_;
  const unsigned exposedAreaVecID = exposedAreaVec_;
  auto* meSCS = meSCS_;
  auto* meFC = meFC_;

  // Zero out nodal fields
  wdist.set_all(ngpMesh, 0.0);
  warea.set_all(ngpMesh, 0.0);

  nalu_ngp::run_face_elem_algorithm(
    meshInfo, faceData_, elemData_, sel,
    KOKKOS_LAMBDA(SimdDataType& fdata) {
      const auto areaOps = nalu_ngp::simd_nodal_field_updater(
        ngpMesh, warea, fdata);
      const auto distOps = nalu_ngp::simd_nodal_field_updater(
        ngpMesh, wdist, fdata);
      const auto dBipOps = nalu_ngp::simd_elem_field_updater(
        ngpMesh, wdistBip, fdata);

      auto& v_coord = fdata.simdElemView.get_scratch_view_2D(coordsID);
      auto& v_area = fdata.simdFaceView.get_scratch_view_2D(exposedAreaVecID);

      const int* faceIpNodeMap = meFC->ipNodeMap();
      for (int ip=0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType aMag = 0.0;
        for (int d=0; d < BcAlgTraits::nDim_; ++d)
          aMag += v_area(ip, d) * v_area(ip, d);
        aMag = stk::math::sqrt(aMag);

        const int nodeR = meSCS->ipNodeMap(fdata.faceOrd)[ip];
        const int nodeL = meSCS->opposingNodes(fdata.faceOrd, ip);

        DoubleType ypBip = 0.0;
        for (int d=0; d < BcAlgTraits::nDim_; ++d) {
          const DoubleType nj = v_area(ip, d) / aMag;
          const DoubleType ej = 0.25 * (v_coord(nodeR, d) - v_coord(nodeL, d));
          ypBip += nj * ej * nj * ej;
        }
        ypBip = stk::math::sqrt(ypBip);

        // Update the wall distance boundary integration pt (Bip)
        dBipOps(ip) = ypBip;

        // Accumulate to the nearest node
        const int ni = faceIpNodeMap[ip];
        distOps(ni, 0) += aMag * ypBip;
        areaOps(ni, 0) += aMag;
      }
    });

  {
    // TODO replace logic with STK NGP parallel sum, but still need to handle
    // periodic the old way
    wdist.modify_on_device();
    wdist.sync_to_host();
    warea.modify_on_device();
    warea.sync_to_host();

    // Synchronize fields for parallel runs
    stk::mesh::FieldBase* wallAreaF = meta.get_field(
      stk::topology::NODE_RANK, "assembled_wall_area_wf");
    stk::mesh::FieldBase* wallDistF = meta.get_field(
      stk::topology::NODE_RANK, "assembled_wall_normal_distance");
    stk::mesh::parallel_sum(realm_.bulk_data(),
                            {wallAreaF, wallDistF});

    if (realm_.hasPeriodic_) {
      const unsigned nComponents = 1;
      const bool bypassFieldChk = false;
      realm_.periodic_field_update(wallAreaF, nComponents, bypassFieldChk);
      realm_.periodic_field_update(wallDistF, nComponents, bypassFieldChk);
    }

    wdist.modify_on_host();
    wdist.sync_to_device();
    warea.modify_on_host();
    warea.sync_to_device();
  }

  sierra::nalu::nalu_ngp::run_entity_algorithm(
    ngpMesh, stk::topology::NODE_RANK, sel,
    KOKKOS_LAMBDA(const MeshIndex& mi) {
      wdist.get(mi, 0) /= warea.get(mi, 0);
    });

  // Indicate that we have modified but don't sync it
  wdist.modify_on_device();
  warea.modify_on_device();
  wdistBip.modify_on_device();
}

INSTANTIATE_KERNEL_FACE_ELEMENT(WallFuncGeometryAlg);

}  // nalu
}  // sierra
