// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef GEOMETRYBOUNDARYALG_H
#define GEOMETRYBOUNDARYALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

/** Compute exposed area vectors for the boundaries
 *
 *  \sa GeometryAlgDriver, GeometryInteriorAlg
 */
template <typename AlgTraits>
class GeometryBoundaryAlg : public Algorithm
{
public:
  GeometryBoundaryAlg(Realm&, stk::mesh::Part*);

  virtual ~GeometryBoundaryAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests dataNeeded_;

  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};

  MasterElement* meSCS_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* GEOMETRYBOUNDARYALG_H */
