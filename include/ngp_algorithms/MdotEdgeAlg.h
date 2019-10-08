/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef MDOTEDGEALG_H
#define MDOTEDGEALG_H

#include "Algorithm.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

class MdotEdgeAlg : public Algorithm
{
public:
  using DblType = double;

  MdotEdgeAlg(Realm&, stk::mesh::Part*);

  virtual ~MdotEdgeAlg() = default;

  void execute() override;

private:
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_ {stk::mesh::InvalidOrdinal};
  unsigned pressure_ {stk::mesh::InvalidOrdinal};
  unsigned densityNp1_ {stk::mesh::InvalidOrdinal};
  unsigned Gpdx_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned Udiag_ {stk::mesh::InvalidOrdinal};
  unsigned massFlowRate_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* MDOTEDGEALG_H */
