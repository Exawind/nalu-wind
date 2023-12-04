// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "user_functions/ZalesakSphereMassFlowRateKernel.h"
#include "EquationSystem.h"
#include "PecletFunction.h"
#include "SolutionOptions.h"
#include "utils/StkHelpers.h"
#include "edge_kernels/EdgeKernelUtils.h"
#include "stk_mesh/base/NgpField.hpp"
#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

ZalesakSphereMassFlowRateEdgeAlg::ZalesakSphereMassFlowRateEdgeAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem,
  const bool useAverages)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  const auto& meta = realm.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  edgeAreaVec_ =
    get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
  massFlowRate_ = get_field_ordinal(
    meta, (useAverages) ? "average_mass_flow_rate" : "mass_flow_rate",
    stk::topology::EDGE_RANK);
}

void
ZalesakSphereMassFlowRateEdgeAlg::execute()
{
  const int ndim = realm_.meta_data().spatial_dimension();

  // Defaults from AMR-Wind
  const double period = 6.0;
  const double xrot = 0.5;
  const double yrot = 0.5;

  // STK stk::mesh::NgpField instances for capture by lambda
  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);
  const auto massFlowRate = fieldMgr.get_field<double>(massFlowRate_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType& /*smdata*/, const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR) {
      // Scratch work array for edgeAreaVector
      NALU_ALIGNED DblType av[NDimMax_];
      // Populate area vector work array
      for (int d = 0; d < ndim; ++d)
        av[d] = edgeAreaVec.get(edge, d);

      NALU_ALIGNED DblType edge_centroid[NDimMax_];
      for (int d = 0; d < ndim; ++d)
        edge_centroid[d] =
          0.5 * coordinates.get(nodeL, d) + 0.5 * coordinates.get(nodeR, d);

      massFlowRate.get(edge, 0) =
        2.0 * M_PI / period *
        ((yrot - edge_centroid[1]) * av[0] + (edge_centroid[0] - xrot) * av[1]);
    });
}

} // namespace nalu
} // namespace sierra
