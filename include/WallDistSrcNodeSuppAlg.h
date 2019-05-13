/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WALLDISTSRCNODESUPPALG_H
#define WALLDISTSRCNODESUPPALG_H

#include "SupplementalAlgorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Entity.hpp"

namespace sierra {
namespace nalu {

class Realm;

class WallDistSrcNodeSuppAlg : public SupplementalAlgorithm
{
public:
  WallDistSrcNodeSuppAlg(Realm&);

  virtual ~WallDistSrcNodeSuppAlg() {}

  virtual void setup() {}

  virtual void node_execute(double*, double*, stk::mesh::Entity);

private:
  WallDistSrcNodeSuppAlg() = delete;
  WallDistSrcNodeSuppAlg(const WallDistSrcNodeSuppAlg&) = delete;

  ScalarFieldType* dualNodalVolume_;
};

}  // nalu
}  // sierra


#endif /* WALLDISTSRCNODESUPPALG_H */
