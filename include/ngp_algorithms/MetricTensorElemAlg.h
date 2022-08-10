// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MetricTensorElemAlg_h
#define MetricTensorElemAlg_h

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class MasterElement;

template <typename AlgTraits>
class MetricTensorElemAlg : public Algorithm
{
public:
  using DblType = double;

  MetricTensorElemAlg(Realm& realm, stk::mesh::Part* part);

  virtual ~MetricTensorElemAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests dataNeeded_;

  unsigned nodalMij_{stk::mesh::InvalidOrdinal};
  unsigned dualNodalVol_{stk::mesh::InvalidOrdinal};

  MasterElement* meSCV_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif
