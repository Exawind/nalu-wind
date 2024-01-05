// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumMassBDFNodeKernel.h"
#include "Realm.h"
#include "utils/FieldHelpers.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

MomentumMassBDFNodeKernel::MomentumMassBDFNodeKernel(
  const stk::mesh::BulkData& bulk)
  : NGPNodeKernel<MomentumMassBDFNodeKernel>(),
    nDim_(bulk.mesh_meta_data().spatial_dimension())
{
  const auto& meta = bulk.mesh_meta_data();

  const auto* velocity =
    meta.get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");

  velocityNID_ =
    velocity->field_of_state(stk::mesh::StateN).mesh_meta_data_ordinal();

  if (velocity->number_of_states() == 2)
    velocityNm1ID_ = velocityNID_;
  else
    velocityNm1ID_ =
      velocity->field_of_state(stk::mesh::StateNM1).mesh_meta_data_ordinal();

  velocityNp1ID_ =
    velocity->field_of_state(stk::mesh::StateNP1).mesh_meta_data_ordinal();

  densityNID_ = get_field_ordinal(meta, "density", stk::mesh::StateN);

  if (velocity->number_of_states() == 2)
    densityNm1ID_ = densityNID_;
  else
    densityNm1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNM1);

  densityNp1ID_ = get_field_ordinal(meta, "density", stk::mesh::StateNP1);

  populate_dnv_states(meta, dnvNm1ID_, dnvNID_, dnvNp1ID_);

  dpdxID_ = get_field_ordinal(meta, "dpdx");
}

void
MomentumMassBDFNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();

  velocityNm1_ = fieldMgr.get_field<double>(velocityNm1ID_);
  velocityN_ = fieldMgr.get_field<double>(velocityNID_);
  velocityNp1_ = fieldMgr.get_field<double>(velocityNp1ID_);
  densityNm1_ = fieldMgr.get_field<double>(densityNm1ID_);
  densityN_ = fieldMgr.get_field<double>(densityNID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);
  dnvNp1_ = fieldMgr.get_field<double>(dnvNp1ID_);
  dnvN_ = fieldMgr.get_field<double>(dnvNID_);
  dnvNm1_ = fieldMgr.get_field<double>(dnvNm1ID_);
  dpdx_ = fieldMgr.get_field<double>(dpdxID_);
  dt_ = realm.get_time_step();
  gamma1_ = realm.get_gamma1();
  gamma2_ = realm.get_gamma2();
  gamma3_ = realm.get_gamma3();
}

KOKKOS_FUNCTION
void
MomentumMassBDFNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  const int nDim = nDim_;

  const NodeKernelTraits::DblType rhoNm1 = densityNm1_.get(node, 0);
  const NodeKernelTraits::DblType rhoN = densityN_.get(node, 0);
  const NodeKernelTraits::DblType rhoNp1 = densityNp1_.get(node, 0);
  const NodeKernelTraits::DblType dnvNp1 = dnvNp1_.get(node, 0);
  const NodeKernelTraits::DblType dnvN = dnvN_.get(node, 0);
  const NodeKernelTraits::DblType dnvNm1 = dnvNm1_.get(node, 0);
  const NodeKernelTraits::DblType lhsfac = gamma1_ * rhoNp1 * dnvNp1 / dt_;
  // deal with lumped mass matrix (diagonal matrix)
  for (int i = 0; i < nDim; ++i) {
    const NodeKernelTraits::DblType uNm1 = velocityNm1_.get(node, i);
    const NodeKernelTraits::DblType uN = velocityN_.get(node, i);
    const NodeKernelTraits::DblType uNp1 = velocityNp1_.get(node, i);
    const NodeKernelTraits::DblType dpdx = dpdx_.get(node, i);

    rhs(i) += -(gamma1_ * rhoNp1 * uNp1 * dnvNp1 + gamma2_ * rhoN * uN * dnvN +
                gamma3_ * rhoNm1 * uNm1 * dnvNm1) /
                dt_ -
              dpdx * dnvNp1;
    lhs(i, i) += lhsfac;
  }
}

} // namespace nalu
} // namespace sierra
