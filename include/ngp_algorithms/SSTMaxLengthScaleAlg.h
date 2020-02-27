// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SSTMAXLENGTHSCALEALG_H
#define SSTMAXLENGTHSCALEALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;

template <typename AlgTraits>
class SSTMaxLengthScaleAlg : public Algorithm
{
public:
  SSTMaxLengthScaleAlg(
    Realm&,
    stk::mesh::Part*);

  virtual ~SSTMaxLengthScaleAlg() = default;

  virtual void execute() override;

private:
  const unsigned maxLengthScale_ {stk::mesh::InvalidOrdinal};
  const unsigned coordinates_    {stk::mesh::InvalidOrdinal};
  MasterElement* meSCS_          {nullptr};
};

}  // nalu
}  // sierra


#endif /* SSTMAXLENGTHSCALEALG_H*/
