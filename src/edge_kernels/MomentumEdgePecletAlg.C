// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <edge_kernels/MomentumEdgePecletAlg.h>
#include <Realm.h>
#include <SolutionOptions.h>
#include <Enums.h>
#include <string>
#include "stk_mesh/base/NgpField.hpp"
#include <ngp_utils/NgpTypes.h>
#include <EquationSystem.h>
#include <ngp_utils/NgpFieldUtils.h>
#include <ngp_utils/NgpLoopUtils.h>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/FieldBLAS.hpp>

namespace sierra {
namespace nalu {

MomentumEdgePecletAlg::MomentumEdgePecletAlg(
  Realm& realm, stk::mesh::Part* part, EquationSystem* eqSystem)
  : Algorithm(realm, part),
    pecletNumber_(get_field_ordinal(
      realm.meta_data(), "peclet_number", stk::topology::EDGE_RANK)),
    pecletFactor_(get_field_ordinal(
      realm.meta_data(), "peclet_factor", stk::topology::EDGE_RANK)),
    density_(get_field_ordinal(realm.meta_data(), "density")),
    viscosity_(get_field_ordinal(
      realm.meta_data(),
      realm.is_turbulent() ? "effective_viscosity_u" : "viscosity")),
    coordinates_(get_field_ordinal(
      realm.meta_data(), realm.solutionOptions_->get_coordinates_name())),
    vrtm_(get_field_ordinal(
      realm.meta_data(), realm.does_mesh_move() ? "velocity_rtm" : "velocity")),
    edgeAreaVec_(get_field_ordinal(
      realm.meta_data(), "edge_area_vector", stk::topology::EDGE_RANK)),
    nDim_(realm.meta_data().spatial_dimension())
{
  const std::string dofName = "velocity";
  pecletFunction_ = eqSystem->ngp_create_peclet_function<double>(dofName);
}

void
MomentumEdgePecletAlg::execute()
{
  using EntityInfoType = nalu_ngp::EntityInfo<stk::mesh::NgpMesh>;
  const auto& meta = realm_.meta_data();
  const auto ngpMesh = realm_.ngp_mesh();
  const auto& fieldMgr = realm_.ngp_field_manager();

  const auto density = fieldMgr.get_field<double>(density_);
  const auto viscosity = fieldMgr.get_field<double>(viscosity_);
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto vrtm = fieldMgr.get_field<double>(vrtm_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  auto pecletNumber = fieldMgr.get_field<double>(pecletNumber_);
  auto pecletFactor = fieldMgr.get_field<double>(pecletFactor_);

  const stk::mesh::Selector sel = meta.locally_owned_part() &
                                  stk::mesh::selectUnion(partVec_) &
                                  !(realm_.get_inactive_selector());

  // pointer for device capture
  auto* pecFunc = pecletFunction_;
  const int ndim = nDim_;
  const auto eps = eps_;

  nalu_ngp::run_edge_algorithm(
    "compute_peclet_factor", ngpMesh, sel,
    KOKKOS_LAMBDA(const EntityInfoType& eInfo) {
      DblType udotx{0.0};

      const auto edge = eInfo.meshIdx;
      const auto& nodes = eInfo.entityNodes;
      const auto nodeL = ngpMesh.fast_mesh_index(nodes[0]);
      const auto nodeR = ngpMesh.fast_mesh_index(nodes[1]);
      const DblType densityL = density.get(nodeL, 0);
      const DblType densityR = density.get(nodeR, 0);

      const DblType viscosityL = viscosity.get(nodeL, 0);
      const DblType viscosityR = viscosity.get(nodeR, 0);

      const DblType diffIp =
        0.5 * (viscosityL / densityL + viscosityR / densityR);

      for (int d = 0; d < ndim; ++d) {
        const DblType dxj =
          coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        udotx += 0.5 * dxj * (vrtm.get(nodeR, d) + vrtm.get(nodeL, d));
      }

      const DblType pecnum = stk::math::abs(udotx) / (diffIp + eps);
      pecletNumber.get(edge, 0) = pecnum;
      pecletFactor.get(edge, 0) = pecFunc->execute(pecnum);
    });
}

void
determine_max_peclet_factor(
  stk::mesh::BulkData& bulk, const stk::mesh::MetaData& meta)
{
  ScalarFieldType* maxPecFac =
    meta.get_field<double>(stk::topology::NODE_RANK, "max_peclet_factor");
  ScalarFieldType* pecletFactor =
    meta.get_field<double>(stk::topology::EDGE_RANK, "peclet_factor");

  stk::mesh::field_fill(0.0, *maxPecFac);

  const stk::mesh::Selector sel =
    stk::mesh::selectField(*pecletFactor) & meta.locally_owned_part();

  for (const auto* ib : bulk.get_buckets(stk::topology::EDGE_RANK, sel)) {
    const auto& b = *ib;
    const size_t length = b.size();
    for (size_t k = 0; k < length; ++k) {
      stk::mesh::Entity edge = b[k];
      const double* pecFac = stk::mesh::field_data(*pecletFactor, edge);
      const auto* nodes = bulk.begin_nodes(edge);
      double* maxPecL = stk::mesh::field_data(*maxPecFac, nodes[0]);
      double* maxPecR = stk::mesh::field_data(*maxPecFac, nodes[1]);
      *maxPecL = std::max(*maxPecL, *pecFac);
      *maxPecR = std::max(*maxPecR, *pecFac);
    }
  }
  stk::mesh::copy_owned_to_shared(bulk, {maxPecFac});
}

void
determine_max_peclet_number(
  stk::mesh::BulkData& bulk, const stk::mesh::MetaData& meta)
{
  ScalarFieldType* maxPecNum =
    meta.get_field<double>(stk::topology::NODE_RANK, "max_peclet_number");
  ScalarFieldType* pecletNumber =
    meta.get_field<double>(stk::topology::EDGE_RANK, "peclet_number");

  stk::mesh::field_fill(0.0, *maxPecNum);

  const stk::mesh::Selector sel =
    stk::mesh::selectField(*pecletNumber) & meta.locally_owned_part();

  for (const auto* ib : bulk.get_buckets(stk::topology::EDGE_RANK, sel)) {
    const auto& b = *ib;
    const size_t length = b.size();
    for (size_t k = 0; k < length; ++k) {
      stk::mesh::Entity edge = b[k];
      const double* pecNum = stk::mesh::field_data(*pecletNumber, edge);
      const auto* nodes = bulk.begin_nodes(edge);
      double* maxPecL = stk::mesh::field_data(*maxPecNum, nodes[0]);
      double* maxPecR = stk::mesh::field_data(*maxPecNum, nodes[1]);
      *maxPecL = std::max(*maxPecL, *pecNum);
      *maxPecR = std::max(*maxPecR, *pecNum);
    }
  }
  stk::mesh::copy_owned_to_shared(bulk, {maxPecNum});
}

} // namespace nalu
} // namespace sierra
