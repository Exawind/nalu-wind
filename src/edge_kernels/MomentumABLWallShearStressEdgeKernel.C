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

template<typename BcAlgTraits>
MomentumABLWallShearStressEdgeKernel<BcAlgTraits>::MomentumABLWallShearStressEdgeKernel(
  stk::mesh::MetaData& meta,
  ElemDataRequests& faceDataPreReqs
) : NGPKernel<MomentumABLWallShearStressEdgeKernel<BcAlgTraits>>(),
    velocityNp1_(get_field_ordinal(meta, "velocity", stk::mesh::StateNP1)),
    bcVelocity_(get_field_ordinal(meta, "wall_velocity_bc")),
    density_(get_field_ordinal(meta, "density")),
    exposedAreaVec_(get_field_ordinal(meta, "exposed_area_vector", meta.side_rank())),
    wallFricVel_(get_field_ordinal(meta, "wall_friction_velocity_bip", meta.side_rank())),
    wallShearStress_(get_field_ordinal(meta, "wall_shear_stress_bip", meta.side_rank())),
    wallNormDist_(get_field_ordinal(meta, "wall_normal_distance_bip", meta.side_rank())),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<BcAlgTraits>())
{
  faceDataPreReqs.add_cvfem_face_me(meFC_);

  faceDataPreReqs.add_gathered_nodal_field(velocityNp1_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(bcVelocity_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_gathered_nodal_field(density_, 1);
  faceDataPreReqs.add_face_field(exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(wallFricVel_, BcAlgTraits::numFaceIp_);
  faceDataPreReqs.add_face_field(wallShearStress_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
  faceDataPreReqs.add_face_field(wallNormDist_, BcAlgTraits::numFaceIp_);
}

template<typename BcAlgTraits>
void
MomentumABLWallShearStressEdgeKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{

  // Unit normal vector
  NALU_ALIGNED DoubleType nx[BcAlgTraits::nDim_];
  NALU_ALIGNED DoubleType tauWall[BcAlgTraits::nDim_];

  const auto& v_vel = scratchViews.get_scratch_view_2D(velocityNp1_);
  const auto& v_bcvel = scratchViews.get_scratch_view_2D(bcVelocity_);
  const auto& v_density = scratchViews.get_scratch_view_1D(density_);
  const auto& v_areavec = scratchViews.get_scratch_view_2D(exposedAreaVec_);
  const auto& v_wallfricvel = scratchViews.get_scratch_view_1D(wallFricVel_);
  const auto& v_wallshearstress = scratchViews.get_scratch_view_2D(wallShearStress_);
  const auto& v_wallnormdist = scratchViews.get_scratch_view_1D(wallNormDist_);

  const int* ipNodeMap = meFC_->ipNodeMap();

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nodeR = ipNodeMap[ip];

    DoubleType amag = 0.0;
    for (int d=0; d < BcAlgTraits::nDim_; ++d) {
      tauWall[d] = v_wallshearstress(ip, d);
      amag += v_areavec(ip, d) * v_areavec(ip, d);
    }
    amag = stk::math::sqrt(amag);

    for (int i=0; i < BcAlgTraits::nDim_; ++i) {
      const int rowR = nodeR * BcAlgTraits::nDim_ + i;
      rhs(rowR) += tauWall[i]*amag;
    }
  }
}

INSTANTIATE_KERNEL_FACE(MomentumABLWallShearStressEdgeKernel)

}  // nalu
}  // sierra
