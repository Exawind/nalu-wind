/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ASSEMBLEWALLDISTEDGESOLVERALGORITHM_H
#define ASSEMBLEWALLDISTEDGESOLVERALGORITHM_H

#include "SolverAlgorithm.h"
#include "FieldTypeDef.h"

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra {
namespace nalu {

class Realm;

class AssembleWallDistEdgeSolverAlgorithm : public SolverAlgorithm
{
public:
  AssembleWallDistEdgeSolverAlgorithm(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*);

  virtual ~AssembleWallDistEdgeSolverAlgorithm() {}

  virtual void initialize_connectivity();

  virtual void execute();

private:
  AssembleWallDistEdgeSolverAlgorithm() = delete;
  AssembleWallDistEdgeSolverAlgorithm(
    const AssembleWallDistEdgeSolverAlgorithm&) = delete;

  VectorFieldType* coordinates_;
  VectorFieldType* edgeAreaVec_;
  VectorFieldType* dphidx_;
};

}  // nalu
}  // sierra


#endif /* ASSEMBLEWALLDISTEDGESOLVERALGORITHM_H */
