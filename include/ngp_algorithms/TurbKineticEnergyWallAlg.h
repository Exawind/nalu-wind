/*------------------------------------------------------------------------*/
/*  Copyright 2019 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TURBKINETICENERGYWALLALG_H
#define TURBKINETICENERGYWALLALG_H 

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

// stk
#include <stk_mesh/base/Types.hpp>

namespace sierra{
namespace nalu{

class MasterElement;

template<typename AlgTraits>
class TurbKineticEnergyWallAlg : public Algorithm
{
public:
  using DblType = double;
  
  TurbKineticEnergyWallAlg( Realm &, stk::mesh::Part *);

  virtual ~TurbKineticEnergyWallAlg() = default;

  void execute();
  void zero_nodal_fields();
  void assemble_nodal_fields();
  void normalize_nodal_fields();
private:
  const DblType cMu_;
  ElemDataRequests dataNeeded_;

  unsigned turbKineticEnergy_           {stk::mesh::InvalidOrdinal};
  unsigned bcTurbKineticEnergy_         {stk::mesh::InvalidOrdinal};
  unsigned bcAssembledTurbKineticEnergy_{stk::mesh::InvalidOrdinal};
  unsigned assembledWallArea_           {stk::mesh::InvalidOrdinal};
  unsigned wallFrictionVelocityBip_     {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_              {stk::mesh::InvalidOrdinal};

  MasterElement* meFC_{nullptr};
};

} // namespace nalu
} // namespace Sierra

#endif
