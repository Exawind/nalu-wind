// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GEOMETRYINTERIORALG_H
#define GEOMETRYINTERIORALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

/** Compute nodal/element volumes and edge area vectors
 *
 *  "edge_area_vector" is only computed if the user has requested edge-based
 *  finite-volume in the input file.
 *
 *  \sa GeometryAlgDriver, GeometryBoundaryAlg
 */
template <typename AlgTraits>
class GeometryInteriorAlg : public Algorithm
{
public:
  GeometryInteriorAlg(Realm&, stk::mesh::Part*);

  virtual ~GeometryInteriorAlg() = default;

  virtual void execute() override;

  void impl_compute_edge_area_vector();
  void impl_compute_dual_nodal_volume();
  void impl_negative_jacobian_check(bool dumpMeshOnFailure = false);

private:
  ElemDataRequests dataNeeded_;

  unsigned dualNodalVol_{stk::mesh::InvalidOrdinal};
  unsigned elemVol_{stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_{stk::mesh::InvalidOrdinal};

  MasterElement* meSCV_{nullptr};
  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* GEOMETRYINTERIORALG_H */
