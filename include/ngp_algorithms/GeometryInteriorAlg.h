/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef GEOMETRYINTERIORALG_H
#define GEOMETRYINTERIORALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

template <typename AlgTraits>
class GeometryInteriorAlg : public Algorithm
{
public:
  GeometryInteriorAlg(
    Realm&,
    stk::mesh::Part*);

  virtual ~GeometryInteriorAlg() = default;

  virtual void execute() override;

private:
  void compute_dual_nodal_volume();

  void compute_edge_area_vector();

  ElemDataRequests dataNeeded_;

  unsigned dualNodalVol_ {stk::mesh::InvalidOrdinal};
  unsigned elemVol_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};

  MasterElement* meSCV_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* GEOMETRYINTERIORALG_H */
