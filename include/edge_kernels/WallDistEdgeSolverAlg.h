/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef WALLDISTEDGESOLVERALG_H
#define WALLDISTEDGESOLVERALG_H

#include "AssembleEdgeSolverAlgorithm.h"

namespace sierra {
namespace nalu {

class WallDistEdgeSolverAlg : public AssembleEdgeSolverAlgorithm
{
public:
  WallDistEdgeSolverAlg(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*);

   virtual ~WallDistEdgeSolverAlg() = default;

   virtual void execute();

 private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra

#endif /* WALLDISTEDGESOLVERALG_H */
