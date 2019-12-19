// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



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
