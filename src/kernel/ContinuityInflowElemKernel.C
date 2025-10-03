// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernel/ContinuityInflowElemKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementRepo.h"
#include "SolutionOptions.h"
#include "TimeIntegrator.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
ContinuityInflowElemKernel<BcAlgTraits>::ContinuityInflowElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  const bool& useShifted,
  ElemDataRequests& dataPreReqs)
  : NGPKernel<ContinuityInflowElemKernel<BcAlgTraits>>(),
    useShifted_(useShifted),
    projTimeScale_(1.0),
    interpTogether_(solnOpts.get_mdot_interp()),
    om_interpTogether_(1.0 - interpTogether_),
    meFC_(
      MasterElementRepo::get_surface_master_element_on_dev(BcAlgTraits::topo_))
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  const std::string velbc_name =
    solnOpts.activateOpenMdotCorrection_ ? "velocity_bc" : "cont_velocity_bc";
  velocityBC_ = get_field_ordinal(metaData, velbc_name);
  densityBC_ = get_field_ordinal(metaData, "density");
  exposedAreaVec_ =
    get_field_ordinal(metaData, "exposed_area_vector", metaData.side_rank());

  // add master elements
  dataPreReqs.add_cvfem_face_me(meFC_);

  // required fields
  dataPreReqs.add_coordinates_field(
    get_field_ordinal(metaData, solnOpts.get_coordinates_name()),
    BcAlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(velocityBC_, BcAlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(densityBC_, 1);
  dataPreReqs.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
}

template <typename BcAlgTraits>
void
ContinuityInflowElemKernel<BcAlgTraits>::setup(
  const TimeIntegrator& timeIntegrator)
{
  const double dt = timeIntegrator.get_time_step();
  const double gamma1 = timeIntegrator.get_gamma1();
  projTimeScale_ = dt / gamma1;
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
ContinuityInflowElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>&,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  DoubleType w_uBip[BcAlgTraits::nDim_];
  DoubleType w_rho_uBip[BcAlgTraits::nDim_];

  const auto& vf_velocityBC = scratchViews.get_scratch_view_2D(velocityBC_);
  const auto& vf_density = scratchViews.get_scratch_view_1D(densityBC_);
  const auto& vf_exposedAreaVec =
    scratchViews.get_scratch_view_2D(exposedAreaVec_);

  const int* ipNodeMap = meFC_->ipNodeMap();
  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {

    // nearest node (to which we will assemble RHS)
    const int nearestNode = ipNodeMap[ip];

    // zero out vector quantities
    for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
      w_uBip[j] = 0.0;
      w_rho_uBip[j] = 0.0;
    }
    DoubleType rhoBip = 0.0;

    for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic) {
      const DoubleType r = shape_fcn<BcAlgTraits, QuadRank::SCV>(
        use_shifted_quad(useShifted_), ip, ic);

      const DoubleType rhoIC = vf_density(ic);
      rhoBip += r * rhoIC;
      for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
        w_uBip[j] += r * vf_velocityBC(ic, j);
        w_rho_uBip[j] += r * rhoIC * vf_velocityBC(ic, j);
      }
    }

    DoubleType mDot = 0.0;
    for (int j = 0; j < BcAlgTraits::nDim_; ++j) {
      mDot += (interpTogether_ * w_rho_uBip[j] +
               om_interpTogether_ * rhoBip * w_uBip[j]) *
              vf_exposedAreaVec(ip, j);
    }

    rhs(nearestNode) -= mDot / projTimeScale_;
  }
}

INSTANTIATE_KERNEL_FACE(ContinuityInflowElemKernel)

} // namespace nalu
} // namespace sierra
