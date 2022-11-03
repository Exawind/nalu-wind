// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "edge_kernels/MomentumABLWallShearStressEdgeKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"
#include "wind_energy/MoninObukhov.h"

#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
MomentumABLWallShearStressEdgeKernel<BcAlgTraits>::
  MomentumABLWallShearStressEdgeKernel(
    bool slip,
    stk::mesh::MetaData& meta,
    ElemDataRequests& faceDataPreReqs,
    ElemDataRequests& elemData)
  : NGPKernel<MomentumABLWallShearStressEdgeKernel<BcAlgTraits>>(),
    slip_(slip),
    exposedAreaVec_(
      get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    wallShearStress_(
      get_field_ordinal(meta, "wall_shear_stress_bip", meta.side_rank())),
    meFC_(MasterElementRepo::get_surface_master_element<
          typename BcAlgTraits::FaceTraits>()),
    meSCS_(MasterElementRepo::get_surface_master_element<
           typename BcAlgTraits::ElemTraits>())
{
  faceDataPreReqs.add_cvfem_face_me(meFC_);
  elemData.add_cvfem_surface_me(meSCS_);

  faceDataPreReqs.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(
    wallShearStress_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
MomentumABLWallShearStressEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& /* lhs */,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews,
  ScratchViews<
    DoubleType,
    DeviceTeamHandleType,
    DeviceShmem>& /* elemScratchViews */,
  int elemFaceOrdinal)
{
  NALU_ALIGNED DoubleType tauWall[BcAlgTraits::nDim_];

  const auto& v_areavec = scratchViews.get_scratch_view_2D(exposedAreaVec_);
  const auto& v_wallshearstress =
    scratchViews.get_scratch_view_2D(wallShearStress_);

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {

    // Decide which node the shear stress will be applied to
    // rowBase is the first row in the matrix associated with the velocity for
    // this node
    const int rowBase =
      slip_ ? (meSCS_->ipNodeMap(elemFaceOrdinal)[ip] * BcAlgTraits::nDim_)
            : (meSCS_->opposingNodes(elemFaceOrdinal, ip) * BcAlgTraits::nDim_);

    DoubleType amag = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d) {
      tauWall[d] = v_wallshearstress(ip, d);
      amag += v_areavec(ip, d) * v_areavec(ip, d);
    }
    amag = stk::math::sqrt(amag);

    for (int i = 0; i < BcAlgTraits::nDim_; ++i) {
      const int myRow = rowBase + i;
      rhs(myRow) += tauWall[i] * amag;
    }
  }
}

INSTANTIATE_KERNEL_FACE_ELEMENT(MomentumABLWallShearStressEdgeKernel)

} // namespace nalu
} // namespace sierra
