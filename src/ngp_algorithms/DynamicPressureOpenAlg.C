// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "SimdInterface.h"
#include "ngp_algorithms/DynamicPressureOpenAlg.h"
#include "BuildTemplates.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "ngp_algorithms/MdotAlgDriver.h"
#include "ngp_algorithms/NgpAlgDriver.h"
#include "ngp_utils/NgpFieldOps.h"
#include "ngp_utils/NgpLoopUtils.h"
#include "ngp_utils/NgpFieldManager.h"
#include "Realm.h"
#include "ScratchViews.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/NgpMesh.hpp"
#include <stk_math/StkMath.hpp>
#include <stk_mesh/base/FieldState.hpp>

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
DynamicPressureOpenAlg<BcAlgTraits>::DynamicPressureOpenAlg(
  Realm& realm, stk::mesh::Part* part)
  : Algorithm(realm, part),
    faceData_(realm.meta_data()),
    density_(
      get_field_ordinal(realm.meta_data(), "density", stk::mesh::StateNP1)),
    velocity_(
      get_field_ordinal(realm.meta_data(), "velocity", stk::mesh::StateNP1)),
    exposedAreaVec_(get_field_ordinal(
      realm.meta_data(), "exposed_area_vector", realm.meta_data().side_rank())),
    openMassFlowRate_(get_field_ordinal(
      realm_.meta_data(),
      "open_mass_flow_rate",
      realm_.meta_data().side_rank())),
    dynPress_(get_field_ordinal(
      realm_.meta_data(), "dynamic_pressure", realm_.meta_data().side_rank())),
    meFC_(
      MasterElementRepo::get_surface_master_element_on_dev(BcAlgTraits::topo_))
{
  faceData_.add_cvfem_face_me(meFC_);
  faceData_.add_coordinates_field(
    get_field_ordinal(realm_.meta_data(), realm_.get_coordinates_name()),
    BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceData_.add_gathered_nodal_field(density_, 1);
  faceData_.add_gathered_nodal_field(velocity_, BcAlgTraits::nDim_);
  faceData_.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceData_.add_face_field(openMassFlowRate_, BcAlgTraits::numFaceIp_);
}

template <typename BcAlgTraits>
void
DynamicPressureOpenAlg<BcAlgTraits>::execute()
{
  using ElemSimdData = nalu_ngp::ElemSimdData<stk::mesh::NgpMesh>;
  const auto& meta = realm_.meta_data();
  const auto& meshInfo = realm_.mesh_info();
  const auto& ngpMesh = meshInfo.ngp_mesh();
  const auto& fieldMgr = meshInfo.ngp_field_manager();

  auto dynPress = fieldMgr.template get_field<double>(dynPress_);
  auto dynPressOps = nalu_ngp::simd_elem_field_updater(ngpMesh, dynPress);

  const stk::mesh::Selector sel =
    meta.locally_owned_part() & stk::mesh::selectUnion(partVec_);

  const std::string algName =
    "DynamicPressureOpenAlg_" + std::to_string(BcAlgTraits::topo_);
  const unsigned areavecID = exposedAreaVec_;
  const unsigned mdotID = openMassFlowRate_;
  const unsigned velID = velocity_;
  const auto useShifted = useShifted_;

  const auto shp =
    shape_fcn<BcAlgTraits, QuadRank::SCV>(use_shifted_quad(useShifted));

  nalu_ngp::run_elem_algorithm(
    algName, meshInfo, realm_.meta_data().side_rank(), faceData_, sel,
    KOKKOS_LAMBDA(ElemSimdData & edata) {
      auto& scrViews = edata.simdScrView;
      const auto& mdot = scrViews.get_scratch_view_1D(mdotID);
      const auto& area = scrViews.get_scratch_view_2D(areavecID);
      const auto& vel = scrViews.get_scratch_view_2D(velID);
      const auto meViews = scrViews.get_me_views(CURRENT_COORDINATES);

      for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
        DoubleType asq = 0.0;
        for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
          const auto av = area(ip, d);
          asq += av * av;
        }
        DoubleType unIp = 0;
        for (int n = 0; n < BcAlgTraits::nodesPerFace_; ++n) {
          const auto r = shp(ip, n);
          for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
            unIp += r * area(ip, d) * vel(n, d);
          }
        }
        unIp /= stk::math::sqrt(asq);
        dynPressOps(edata, ip) = stk::math::if_then_else(
          mdot(ip) < 0, 0.5 * stk::math::abs(mdot(ip) * unIp), 0);
      }
    });
  dynPress.modify_on_device();
}

INSTANTIATE_KERNEL_FACE(DynamicPressureOpenAlg)

} // namespace nalu
} // namespace sierra
