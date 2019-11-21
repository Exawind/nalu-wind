// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#include "edge_kernels/WallDistEdgeSolverAlg.h"
#include "utils/StkHelpers.h"

namespace sierra {
namespace nalu {

WallDistEdgeSolverAlg::WallDistEdgeSolverAlg(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem)
  : AssembleEdgeSolverAlgorithm(realm, part, eqSystem)
{
  auto& meta = realm_.meta_data();

  coordinates_ = get_field_ordinal(meta, realm.get_coordinates_name());
  edgeAreaVec_ = get_field_ordinal(meta, "edge_area_vector", stk::topology::EDGE_RANK);
}

void
WallDistEdgeSolverAlg::execute()
{
  const int ndim = realm_.meta_data().spatial_dimension();

  const auto& fieldMgr = realm_.ngp_field_manager();
  const auto coordinates = fieldMgr.get_field<double>(coordinates_);
  const auto edgeAreaVec = fieldMgr.get_field<double>(edgeAreaVec_);

  run_algorithm(
    realm_.bulk_data(),
    KOKKOS_LAMBDA(
      ShmemDataType& smdata,
      const stk::mesh::FastMeshIndex& edge,
      const stk::mesh::FastMeshIndex& nodeL,
      const stk::mesh::FastMeshIndex& nodeR)
    {
      double asq = 0.0;
      double axdx = 0.0;
      for (int d=0; d<ndim; d++) {
        const double axj = edgeAreaVec.get(edge, d);
        const double dxj = coordinates.get(nodeR, d) - coordinates.get(nodeL, d);
        asq += axj * axj;
        axdx += axj * dxj;
      }

      double pfac = 1.0;
      const double lhsfac = pfac * asq / axdx;

      // Left node
      smdata.lhs(0, 0) = +lhsfac;
      smdata.lhs(0, 1) = -lhsfac;

      // Right node
      smdata.lhs(1,0) = -lhsfac;
      smdata.lhs(1,1) = +lhsfac;

      // No RHS contributions
    });
}

}  // nalu
}  // sierra
