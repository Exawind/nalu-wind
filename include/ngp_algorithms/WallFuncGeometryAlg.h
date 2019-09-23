/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
