// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "kernel/ScalarFaceFluxBCElemKernel.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"

#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

#include "stk_mesh/base/Field.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
ScalarFaceFluxBCElemKernel<BcAlgTraits>::ScalarFaceFluxBCElemKernel(
  const stk::mesh::BulkData& bulk,
  GenericFieldType* scalarQ,
  std::string coordsName,
  bool useShifted,
  ElemDataRequests& faceDataPreReqs)
  : NGPKernel<ScalarFaceFluxBCElemKernel<BcAlgTraits>>(),
    coordinates_(get_field_ordinal(bulk.mesh_meta_data(), coordsName)),
    bcScalarQ_(scalarQ->mesh_meta_data_ordinal()),
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
  faceDataPreReqs.add_face_field(bcScalarQ_, BcAlgTraits::numFaceIp_);
  faceDataPreReqs.add_face_field(
    exposedAreaVec_, BcAlgTraits::numFaceIp_, BcAlgTraits::nDim_);
}

template <typename BcAlgTraits>
KOKKOS_FUNCTION void
ScalarFaceFluxBCElemKernel<BcAlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>&,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const auto& v_bcQ = scratchViews.get_scratch_view_1D(bcScalarQ_);
  const auto& v_areav = scratchViews.get_scratch_view_2D(exposedAreaVec_);

  const int* ipNodeMap = meFC_->ipNodeMap();

  for (int ip = 0; ip < BcAlgTraits::numFaceIp_; ++ip) {
    const int nodeR = ipNodeMap[ip];

    // Compute magnitude of area vector at this integration point
    DoubleType aMag = 0.0;
    for (int d = 0; d < BcAlgTraits::nDim_; ++d)
      aMag += v_areav(ip, d) * v_areav(ip, d);
    aMag = stk::math::sqrt(aMag);

    // Get the flux at this integration point.
    DoubleType fluxBip = v_bcQ(ip);

    rhs(nodeR) += fluxBip * aMag;
  }
}

INSTANTIATE_KERNEL_FACE(ScalarFaceFluxBCElemKernel)

} // namespace nalu
} // namespace sierra
