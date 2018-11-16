/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#include "AssembleWallDistEdgeSolverAlgorithm.h"

#include "WallDistEquationSystem.h"
#include "EquationSystem.h"
#include "LinearSystem.h"
#include "Realm.h"

#include "stk_mesh/base/Part.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/BulkData.hpp"

#include <vector>
#include <cmath>

namespace sierra {
namespace nalu {

AssembleWallDistEdgeSolverAlgorithm::AssembleWallDistEdgeSolverAlgorithm(
  Realm& realm,
  stk::mesh::Part* part,
  EquationSystem* eqSystem)
  : SolverAlgorithm(realm, part, eqSystem)
{
  auto& meta = realm_.meta_data();

  coordinates_ = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, realm_.get_coordinates_name());
  edgeAreaVec_ = meta.get_field<VectorFieldType>(
    stk::topology::EDGE_RANK, "edge_area_vector");
  dphidx_ = meta.get_field<VectorFieldType>(
    stk::topology::NODE_RANK, "dwalldistdx");
}

void
AssembleWallDistEdgeSolverAlgorithm::initialize_connectivity()
{
  eqSystem_->linsys_->buildEdgeToNodeGraph(partVec_);
}

void
AssembleWallDistEdgeSolverAlgorithm::execute()
{
  auto& meta = realm_.meta_data();
  auto& bulk = realm_.bulk_data();
  const int nDim = meta.spatial_dimension();

  // WallDistEquationSystem* wdEqs = reinterpret_cast<WallDistEquationSystem*>(eqSystem_);
  // const int pValue = wdEqs->pValue();

  std::vector<double> lhs(4);
  std::vector<double> rhs(2, 0.0);
  std::vector<int> scratchIds(2);
  std::vector<double> scratchVals(2);
  std::vector<stk::mesh::Entity> connected_nodes(2);
  std::vector<double> areaVec(nDim);

  double* p_lhs = lhs.data();
  double* p_areaVec = areaVec.data();

  stk::mesh::Selector sel = meta.locally_owned_part()
    & stk::mesh::selectUnion(partVec_)
    & !(realm_.get_inactive_selector());

  const auto& bkts = bulk.get_buckets(stk::topology::EDGE_RANK, sel);

  for (auto b: bkts) {
    const double* av = stk::mesh::field_data(*edgeAreaVec_, *b);

    for (size_t k=0; k < b->size(); k++) {
      ThrowAssert(b->num_nodes(k) == 2);

      const auto* edge_node_rels = b->begin_nodes(k);

      for (int j=0; j<nDim; j++)
        p_areaVec[j] = av[k*nDim + j];

      auto nodeL = edge_node_rels[0];
      auto nodeR = edge_node_rels[1];

      connected_nodes[0] = nodeL;
      connected_nodes[1] = nodeR;

      const double* coordL = stk::mesh::field_data(*coordinates_, nodeL);
      const double* coordR = stk::mesh::field_data(*coordinates_, nodeR);

      double asq = 0.0;
      double axdx = 0.0;
      for (int j=0; j<nDim; j++) {
        const double axj = p_areaVec[j];
        const double dxj = coordR[j] - coordL[j];
        asq += axj * axj;
        axdx += axj * dxj;
      }

      double pfac = 1.0;
      // if (pValue > 2) {
      //   double gjipsum = 0.0;
      //   const double* GpdxL = stk::mesh::field_data(*dphidx_, nodeL);
      //   const double* GpdxR = stk::mesh::field_data(*dphidx_, nodeR);
      //   for (int j=0; j < nDim; j++) {
      //     double gjip = 0.5 * (GpdxL[j] + GpdxR[j]);
      //     gjipsum += gjip * gjip;
      //   }

      //   pfac = std::pow(gjipsum, (pValue - 2)/2);
      // }

      const double lhsfac = pfac * asq / axdx;

      // Left node
      p_lhs[0] = lhsfac;
      p_lhs[1] = -lhsfac;

      // Right node
      p_lhs[2] = -lhsfac;
      p_lhs[3] = lhsfac;

      // No RHS contributions

      apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
    }
  }
}

}  // nalu
}  // sierra
