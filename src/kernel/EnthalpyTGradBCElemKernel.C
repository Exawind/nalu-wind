// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernel/EnthalpyTGradBCElemKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
EnthalpyTGradBCElemKernel<BcAlgTraits>::EnthalpyTGradBCElemKernel(
  const stk::mesh::BulkData& bulk,
  ScalarFieldType* bcTGrad,
  ScalarFieldType* evisc,
  ScalarFieldType* specificHeat,
  std::string coordsName,
  bool useShifted,
  ElemDataRequests& faceDataPreReqs)
  : NGPKernel<EnthalpyTGradBCElemKernel<BcAlgTraits>>(),
    coordinates_(get_field_ordinal(bulk.mesh_meta_data(), coordsName)),
    bcTGrad_(bcTGrad->mesh_meta_data_ordinal()),
    evisc_(evisc->mesh_meta_data_ordinal()),
    specificHeat_(specificHeat->mesh_meta_data_ordinal()),
    exposedAreaVec_(get_field_ordinal(
      bulk.mesh_meta_data(),
      "exposed_area_vector",
      bulk.mesh_meta_data().side_rank())),
    useShifted_(useShifted),
    meFC_(sierra::nalu::MasterElementRepo::get_surface_master_element<
          BcAlgTraits>())
{
  // Register necessary data for use in execute method
  faceDataPreReqs.add_cvfem_face_me(meFC_);

  faceDataPreReqs.add_coordinates_field(
    coordinates_, BcAlgTraits::nDim_, CURRENT_COORDINATES);
  faceDataPreReqs.add_gathered_nodal_field(bcTGrad_, 1);
  faceDataPreReqs.add_gathered_nodal_field(evisc_, 1);
  faceDataPreReqs.add_gathered_nodal_field(specificHeat_, 1);
  faceDataPreReqs.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);

  faceDataPreReqs.add_master_element_call(
    (useShifted_ ? FC_SHIFTED_SHAPE_FCN : FC_SHAPE_FCN), CURRENT_COORDINATES);
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
EnthalpyTGradBCElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>&,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const auto& v_tgrad = scratchViews.get_scratch_view_1D(bcTGrad_);
  const auto& v_visc = scratchViews.get_scratch_view_1D(evisc_);
  const auto& v_Cp = scratchViews.get_scratch_view_1D(specificHeat_);
  const auto& v_areavec = scratchViews.get_scratch_view_2D(exposedAreaVec_);

  const auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  const auto& v_shape_fcn =
    useShifted_ ? meViews.fc_shifted_shape_fcn : meViews.fc_shape_fcn;

  const int* ipNodeMap = meFC_->ipNodeMap();

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nearestNode = ipNodeMap[ip];

    // Compute magnitude of area vector at this integration point
    DoubleType aMag = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d)
      aMag += v_areavec(ip, d) * v_areavec(ip, d);
    aMag = stk::math::sqrt(aMag);

    // Interpolate desired data to the face integration points
    DoubleType tgradBip = 0.0;
    DoubleType viscBip = 0.0;
    DoubleType cpBip = 0.0;
    for (int ic = 0; ic < BcAlgTraits::nodesPerFace_; ++ic) {
      const DoubleType r = v_shape_fcn(ip, ic);
      tgradBip += r * v_tgrad(ic);
      viscBip += r * v_visc(ic);
      cpBip += r * v_Cp(ic);
    }

    // heat flux = mu_eff * Cp * dT/dx * areaVec
    // Area vector points into the domain, so negate the flux direction
    rhs(nearestNode) -= viscBip * cpBip * tgradBip * aMag;
  }
}

INSTANTIATE_KERNEL_FACE(EnthalpyTGradBCElemKernel)

} // namespace nalu
} // namespace sierra
