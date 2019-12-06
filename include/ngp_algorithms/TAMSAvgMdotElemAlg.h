// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
