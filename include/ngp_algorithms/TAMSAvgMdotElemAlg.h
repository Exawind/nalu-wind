/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TAMSAVGMDOTELEMALG_H
#define TAMSAVGMDOTELEMALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MasterElement;

template<typename AlgTraits>
class TAMSAvgMdotElemAlg : public Algorithm
{

public:
  using DblType = double;

  TAMSAvgMdotElemAlg(
    Realm&,
    stk::mesh::Part*);

  virtual ~TAMSAvgMdotElemAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests dataNeeded_;

  unsigned avgTime_{stk::mesh::InvalidOrdinal};
  unsigned mdot_{stk::mesh::InvalidOrdinal};
  unsigned avgMdot_ {stk::mesh::InvalidOrdinal};

  const bool useShifted_{false};

  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* TAMSAVGMDOTELEMALG_H */
