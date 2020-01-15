// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "kernel/MomentumBuoyancySrcElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "SolutionOptions.h"

// template and scratch space
#include "BuildTemplates.h"
#include "ScratchViews.h"
#include "utils/StkHelpers.h"

// stk_mesh/base/fem
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>

namespace sierra {
namespace nalu {

template<class AlgTraits>
MomentumBuoyancySrcElemKernel<AlgTraits>::MomentumBuoyancySrcElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs)
  : rhoRef_(solnOpts.referenceDensity_)
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());
  
  const std::vector<double>& solnOptsGravity = solnOpts.get_gravity_vector(AlgTraits::nDim_);
  for (int i = 0; i < AlgTraits::nDim_; i++)
    gravity_[i] = solnOptsGravity[i];

  meSCV_ = MasterElementRepo::get_volume_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(SCV_SHAPE_FCN, CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
MomentumBuoyancySrcElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& /* lhs */,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const auto& v_densityNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  const auto& v_scv_volume = scratchViews.get_me_views(CURRENT_COORDINATES).scv_volume;
  const auto& v_shape_function = scratchViews.get_me_views(CURRENT_COORDINATES).scv_shape_fcn;
  const auto* ipNodeMap = meSCV_->ipNodeMap();

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap[ip];
    DoubleType rhoNp1 = 0.0;

    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function(ip, ic);
      rhoNp1 += r * v_densityNp1(ic);
    }

    // Compute RHS
    const DoubleType scV = v_scv_volume(ip);
    const int nnNdim = nearestNode * AlgTraits::nDim_;
    const DoubleType fac = (rhoNp1 - rhoRef_) * scV;
    for (int j=0; j < AlgTraits::nDim_; j++) {
      rhs(nnNdim + j) += fac * gravity_[j];
    }

    // No LHS contributions
  }
}

INSTANTIATE_KERNEL(MomentumBuoyancySrcElemKernel)

}  // nalu
}  // sierra
