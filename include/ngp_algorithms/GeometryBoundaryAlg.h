/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
  GeometryBoundaryAlg(
    Realm&,
    stk::mesh::Part*);

  virtual ~GeometryBoundaryAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests dataNeeded_;

  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};

  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* GEOMETRYBOUNDARYALG_H */
