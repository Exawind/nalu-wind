// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef WALLFUNCGEOMETRYALG_H
#define WALLFUNCGEOMETRYALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

template <typename BcAlgTraits>
class WallFuncGeometryAlg : public Algorithm
{
public:
  using DblType = double;

  WallFuncGeometryAlg(Realm&, stk::mesh::Part*);

  virtual ~WallFuncGeometryAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests faceData_;
  ElemDataRequests elemData_;

  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned wallNormDistBip_ {stk::mesh::InvalidOrdinal};
  unsigned wallArea_ {stk::mesh::InvalidOrdinal};
  unsigned wallNormDist_ {stk::mesh::InvalidOrdinal};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* WALLFUNCGEOMETRYALG_H */
