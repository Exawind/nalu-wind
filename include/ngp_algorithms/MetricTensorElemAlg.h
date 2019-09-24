/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
