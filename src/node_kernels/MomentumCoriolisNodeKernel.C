// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumCoriolisNodeKernel.h"
#include "Realm.h"
#include "utils/StkHelpers.h"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

MomentumCoriolisNodeKernel::MomentumCoriolisNodeKernel(
  const stk::mesh::BulkData& bulk, const SolutionOptions& solnOpts)
  : NGPNodeKernel<MomentumCoriolisNodeKernel>(), cor_(solnOpts)
{
  const auto& meta = bulk.mesh_meta_data();

  velocityNp1ID_ = get_field_ordinal(meta, "velocity");
  densityNp1ID_ = get_field_ordinal(meta, "density");
  dualNodalVolumeID_ = get_field_ordinal(meta, "dual_nodal_volume");
}

void
MomentumCoriolisNodeKernel::setup(Realm& realm)
{
  const auto& fieldMgr = realm.ngp_field_manager();
  dualNodalVolume_ = fieldMgr.get_field<double>(dualNodalVolumeID_);
  velocityNp1_ = fieldMgr.get_field<double>(velocityNp1ID_);
  densityNp1_ = fieldMgr.get_field<double>(densityNp1ID_);
}

KOKKOS_FUNCTION
void
MomentumCoriolisNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  NALU_ALIGNED NodeKernelTraits::DblType vel[NodeKernelTraits::NDimMax];
  NodeKernelTraits::DblType rhoNp1 = densityNp1_.get(node, 0);
  NodeKernelTraits::DblType dualVol = dualNodalVolume_.get(node, 0);

  for (int i = 0; i < NodeKernelTraits::NDimMax; ++i) {
    vel[i] = velocityNp1_.get(node, i);
  }

  // calculate the velocity vector in east-north-up coordinates
  const NodeKernelTraits::DblType ue = cor_.eastVector_[0] * vel[0] +
                                       cor_.eastVector_[1] * vel[1] +
                                       cor_.eastVector_[2] * vel[2];
  const NodeKernelTraits::DblType un = cor_.northVector_[0] * vel[0] +
                                       cor_.northVector_[1] * vel[1] +
                                       cor_.northVector_[2] * vel[2];
  const NodeKernelTraits::DblType uu = cor_.upVector_[0] * vel[0] +
                                       cor_.upVector_[1] * vel[1] +
                                       cor_.upVector_[2] * vel[2];

  // calculate acceleration in east-north-up coordinates
  const NodeKernelTraits::DblType ae =
    cor_.corfac_ * (un * cor_.sinphi_ - uu * cor_.cosphi_);
  const NodeKernelTraits::DblType an = -cor_.corfac_ * ue * cor_.sinphi_;
  const NodeKernelTraits::DblType au = cor_.corfac_ * ue * cor_.cosphi_;

  // calculate acceleration in model x-y-z coordinates
  const NodeKernelTraits::DblType ax = ae * cor_.eastVector_[0] +
                                       an * cor_.northVector_[0] +
                                       au * cor_.upVector_[0];
  const NodeKernelTraits::DblType ay = ae * cor_.eastVector_[1] +
                                       an * cor_.northVector_[1] +
                                       au * cor_.upVector_[1];
  const NodeKernelTraits::DblType az = ae * cor_.eastVector_[2] +
                                       an * cor_.northVector_[2] +
                                       au * cor_.upVector_[2];

  const double fac2 = rhoNp1 * dualVol;
  rhs(0) += fac2 * ax;
  rhs(1) += fac2 * ay;
  rhs(2) += fac2 * az;

  // Only the off-diagonal LHS entries are non-zero
  lhs(0, 1) += fac2 * cor_.Jxy_;
  lhs(0, 2) += fac2 * cor_.Jxz_;
  lhs(1, 0) -= fac2 * cor_.Jxy_; // Jyx = - Jxy
  lhs(1, 2) += fac2 * cor_.Jyz_;
  lhs(2, 0) -= fac2 * cor_.Jxz_; // Jzx = - Jxz
  lhs(2, 1) -= fac2 * cor_.Jyz_; // Jzy = - Jyz
}

} // namespace nalu
} // namespace sierra
