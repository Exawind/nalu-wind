/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "kernel/MomentumMassElemKernel.h"
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
MomentumMassElemKernel<AlgTraits>::MomentumMassElemKernel(
  const stk::mesh::BulkData& bulkData,
  const SolutionOptions& solnOpts,
  ElemDataRequests& dataPreReqs,
  const bool lumpedMass)
  : lumpedMass_(lumpedMass)
{
  const stk::mesh::MetaData& metaData = bulkData.mesh_meta_data();
  VectorFieldType *velocity = metaData.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "velocity");
  ScalarFieldType *density = metaData.get_field<ScalarFieldType>(
    stk::topology::NODE_RANK, "density");

  velocityN_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateN);
  velocityNp1_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateNP1);
  if (velocity->number_of_states() == 2)
    velocityNm1_ = velocityN_;
  else
    velocityNm1_ = get_field_ordinal(metaData, "velocity", stk::mesh::StateNM1);

  densityN_ = get_field_ordinal(metaData, "density", stk::mesh::StateN);
  densityNp1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNP1);
  if (density->number_of_states() == 2)
    densityNm1_ = densityN_;
  else
    densityNm1_ = get_field_ordinal(metaData, "density", stk::mesh::StateNM1);

  Gjp_ = get_field_ordinal(metaData, "dpdx");
  coordinates_ = get_field_ordinal(metaData, solnOpts.get_coordinates_name());

  meSCV_ = sierra::nalu::MasterElementRepo::get_volume_master_element<AlgTraits>();

  // add master elements
  dataPreReqs.add_cvfem_volume_me(meSCV_);

  // fields and data
  dataPreReqs.add_coordinates_field(coordinates_, AlgTraits::nDim_, CURRENT_COORDINATES);
  dataPreReqs.add_gathered_nodal_field(densityNm1_, 1);
  dataPreReqs.add_gathered_nodal_field(densityN_, 1);
  dataPreReqs.add_gathered_nodal_field(densityNp1_, 1);
  dataPreReqs.add_gathered_nodal_field(velocityNm1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(velocityN_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(velocityNp1_, AlgTraits::nDim_);
  dataPreReqs.add_gathered_nodal_field(Gjp_, AlgTraits::nDim_);
  dataPreReqs.add_master_element_call(SCV_VOLUME, CURRENT_COORDINATES);
  dataPreReqs.add_master_element_call(
    (lumpedMass_ ? SCV_SHIFTED_SHAPE_FCN : SCV_SHAPE_FCN), CURRENT_COORDINATES);

  const std::string dofName = "velocity";
  diagRelaxFactor_ = solnOpts.get_relaxation_factor(dofName);
}

template<typename AlgTraits>
void
MomentumMassElemKernel<AlgTraits>::setup(const TimeIntegrator& timeIntegrator)
{
  dt_ = timeIntegrator.get_time_step();
  gamma1_ = timeIntegrator.get_gamma1();
  gamma2_ = timeIntegrator.get_gamma2();
  gamma3_ = timeIntegrator.get_gamma3(); // gamma3 may be zero
}

template<typename AlgTraits>
void
MomentumMassElemKernel<AlgTraits>::execute(
  SharedMemView<DoubleType**, DeviceShmem>& lhs,
  SharedMemView<DoubleType*, DeviceShmem>& rhs,
  ScratchViews<DoubleType, DeviceTeamHandleType, DeviceShmem>& scratchViews)
{
  NALU_ALIGNED DoubleType w_uNm1 [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uN   [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_uNp1 [AlgTraits::nDim_];
  NALU_ALIGNED DoubleType w_Gjp  [AlgTraits::nDim_];

  auto& v_densityNm1 = scratchViews.get_scratch_view_1D(densityNm1_);
  auto& v_densityN = scratchViews.get_scratch_view_1D(densityN_);
  auto& v_densityNp1 = scratchViews.get_scratch_view_1D(densityNp1_);
  auto& v_velocityNm1 = scratchViews.get_scratch_view_2D(velocityNm1_);
  auto& v_velocityN = scratchViews.get_scratch_view_2D(velocityN_);
  auto& v_velocityNp1 = scratchViews.get_scratch_view_2D(velocityNp1_);
  auto& v_Gpdx = scratchViews.get_scratch_view_2D(Gjp_);

  auto& meViews = scratchViews.get_me_views(CURRENT_COORDINATES);
  auto& v_scv_volume = meViews.scv_volume;
  auto& v_shape_function = lumpedMass_ ? meViews.scv_shifted_shape_fcn : meViews.scv_shape_fcn;

  const int* ipNodeMap = meSCV_->ipNodeMap();

  for (int ip=0; ip < AlgTraits::numScvIp_; ++ip) {
    const int nearestNode = ipNodeMap[ip];

    DoubleType rhoNm1 = 0.0;
    DoubleType rhoN   = 0.0;
    DoubleType rhoNp1 = 0.0;
    for (int j=0; j < AlgTraits::nDim_; j++) {
      w_uNm1[j] = 0.0;
      w_uN[j] = 0.0;
      w_uNp1[j] = 0.0;
      w_Gjp[j] = 0.0;
    }

    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const DoubleType r = v_shape_function(ip, ic);

      rhoNm1 += r * v_densityNm1(ic);
      rhoN   += r * v_densityN(ic);
      rhoNp1 += r * v_densityNp1(ic);
      for (int j=0; j < AlgTraits::nDim_; j++) {
        w_uNm1[j] += r * v_velocityNm1(ic, j);
        w_uN[j]   += r * v_velocityN(ic, j);
        w_uNp1[j] += r * v_velocityNp1(ic, j);
        w_Gjp[j]  += r * v_Gpdx(ic, j);
      }
    }

    const DoubleType scV = v_scv_volume(ip);
    const int nnNdim = nearestNode * AlgTraits::nDim_;
    // Compute RHS
    for (int j=0; j < AlgTraits::nDim_; ++j) {
      rhs(nnNdim + j) +=
        - ( gamma1_ * rhoNp1 * w_uNp1[j] +
            gamma2_ * rhoN   * w_uN[j] +
            gamma3_ * rhoNm1 * w_uNm1[j]) * scV / dt_
        - w_Gjp[j] * scV;
    }

    // Compute LHS
    for (int ic=0; ic < AlgTraits::nodesPerElement_; ++ic) {
      const int icNdim = ic * AlgTraits::nDim_;
      const DoubleType r = v_shape_function(ip, ic);
      const DoubleType lhsfac = r * gamma1_ * rhoNp1 * scV / dt_ * diagRelaxFactor_;

      for (int j=0; j<AlgTraits::nDim_; ++j) {
        const int indexNN = nnNdim + j;
        lhs(indexNN,icNdim+j) += lhsfac;
      }
    }
  }
}

INSTANTIATE_KERNEL(MomentumMassElemKernel)

}  // nalu
}  // sierra
