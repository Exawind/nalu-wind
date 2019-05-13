/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleOversetWallDistAlgorithm_h
#define AssembleOversetWallDistAlgorithm_h

#include "overset/OversetConstraintBase.h"

namespace stk {
namespace mesh {
class Part;
class FieldBase;
}
}

namespace sierra{
namespace nalu{

class Realm;

/** Populate the Continuity linear system for the overset constraint rows
 *
 *  This algorithm will reset the fringe rows for the pressure Poisson solve and
 *  populate them with the interpolation coefficients based on the donor
 *  element. The constraint equation is scaled by the projection time scale and
 *  dual nodal volume to make this row well behaved compared to its neighbors.
 *
 */
class AssembleOversetWallDistAlgorithm : public OversetConstraintBase
{
public:

  AssembleOversetWallDistAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    stk::mesh::FieldBase *fieldQ);

  virtual ~AssembleOversetWallDistAlgorithm() = default;

  virtual void execute();
};

} // namespace nalu
} // namespace Sierra

#endif
