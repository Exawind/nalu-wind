/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/ContinuityMassElemKernel.h"
#include "AlgTraits.h"
#include "master_element/MasterElement.h"
#include "master_element/MasterElementFactory.h"
#include "TimeIntegrator.h"
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

template<typename AlgTraits>
ContinuityMassElemKernel<AlgTraits>::ContinuityMassElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs,
  const bool lumpedMass)
  : lumpedMass_(lumpedMass)
{
  // save off fields
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();

  ScalarFieldType* density = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");
  densityN_ = density->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal();
  densityNp1_ = density->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();
  if (density->number_of_states() == 2)
    densityNm1_ = densityN_;
  else
    densityNm1_ = density->field_of_state(stk::mesh::StateNM1).mesh_meta_data_ordinal();
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  meSCV_ = sierra::nalu::MasterElementRepo::get_volume_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(densityNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(densityN_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);

  dataPreReqs.add_master_element_call(
    (lumpedMass_ ? SCV_SHIFTED_SHAPE_FCN : SCV_SHAPE_FCN), CURRENT_COORDINATES);
}

template<typename AlgTraits>
void
ContinuityMassElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3(); // gamma3 may be zero
}

template<typename AlgTraits>
void
ContinuityMassElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>&/*lhs*/,
  SharedMemView<DoubleType*, DeviceShmem>&rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  const DoubleType projTimeScale = dt_/gamma1_;

  auto& v_densityNm1 = scratchViews.get_scratch_view_1D(densityNm1_);
  auto& v_densityN = scratchViews.get_scratch_view_1D(densityN_);
  auto& v_densityNp1 = scratchViews.get_scratch_view_1D(densityNp1_);

  auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  auto& v_scv_volume = meViews.scv_volume;
  auto& v_shape_function = lumpedMass_ ? meViews.scv_shifted_shape_fcn : meViews.scv_shape_fcn;

  const int* ipNodeMap = meSCV_->ipNodeMap();


  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap[ip];

    DoubleType rhoNm1 = 0.0;
    DoubleType rhoN   = 0.0;
    DoubleType rhoNp1 = 0.0;
    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function(ip, ic);

      rhoNm1 += r * v_densityNm1(ic);
      rhoN   += r * v_densityN(ic);
      rhoNp1 += r * v_densityNp1(ic);
    }

    const DoubleType scV = v_scv_volume(ip);
    rhs(nearestNode) += - ( gamma1_ * rhoNp1 + gamma2_ * rhoN +
                            gamma3_ * rhoNm1 ) * scV / dt_ / projTimeScale;

    // manage LHS : N/A
  }
}

INSTANTIATE_KERNEL(ContinuityMassElemKernel)

}  // nalu
}  // sierra
