// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "node_kernels/MomentumRayleighDampingNodeKernel.h"
#include "Realm.h"

#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Types.hpp"
#include "utils/StkHelpers.h"

#include "SolutionOptions.h"

#include "WallDistEquationSystem.h"

namespace sierra {
namespace nalu {

template <typename T = double>
stk::mesh::NgpField<T>&
get_ngp_field(
  const stk::mesh::MetaData& meta,
  std::string name,
  stk::mesh::FieldState state = stk::mesh::StateNP1)
{
  ThrowRequire(meta.get_field(stk::topology::NODE_RANK, name));
  ThrowRequire(
    meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
  return stk::mesh::get_updated_ngp_field<T>(
    *meta.get_field(stk::topology::NODE_RANK, name)->field_state(state));
}

MomentumRayleighDampingNodeKernel::MomentumRayleighDampingNodeKernel(
  const stk::mesh::MetaData& meta,
  RayleighDampingParameters params,
  std::string name)
  : NGPNodeKernel<MomentumRayleighDampingNodeKernel>(),
    nDim_(meta.spatial_dimension()),
    params_(params)
{
  velocity_ = get_ngp_field(meta, "velocity", stk::mesh::StateNP1);
  volume_ = get_ngp_field(meta, "dual_nodal_volume", stk::mesh::StateNP1);
  auto wall_dist_name = WallDistEquationSystem::min_wall_distance_name(name);
  distance_ = get_ngp_field(meta, wall_dist_name);
}

void
MomentumRayleighDampingNodeKernel::execute(
  NodeKernelTraits::LhsType& lhs,
  NodeKernelTraits::RhsType& rhs,
  const stk::mesh::FastMeshIndex& node)
{
  double c = 0;
  if (distance_.get(node, 0) < params_.width) {
    c = 0.5 * params_.cmax *
        (1 + stk::math::cos(M_PI * distance_.get(node, 0) / params_.width));
  }

  Kokkos::Array<ftype, max_dim> cell_damping_force;
  for (int d = 0; d < 3; ++d) {
    cell_damping_force[d] =
      c * volume_.get(node, 0) * (params_.uref[d] - velocity_.get(node, d));
  }
  // derivstive of cell damping wrt u_j
  // just a constant, diagonal matrix
  const ftype dfiduj = -c * volume_.get(node, 0);

  for (int d = 0; d < nDim_; ++d) {
    lhs(d, d) -= dfiduj;
    rhs(d) += cell_damping_force[d];
  }
}

} // namespace nalu
} // namespace sierra
