// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DynamicPressureOpenAlg_H
#define DynamicPressureOpenAlg_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class Realm;

template<typename BcAlgTraits>
class DynamicPressureOpenAlg : public Algorithm
{
public:
  DynamicPressureOpenAlg(
    Realm&,
    stk::mesh::Part*);

  virtual ~DynamicPressureOpenAlg() = default;
  virtual void execute() override;

private:
  ElemDataRequests faceData_;
  unsigned density_         {stk::mesh::InvalidOrdinal};
  unsigned velocity_         {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_   {stk::mesh::InvalidOrdinal};
  unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
  unsigned dynPress_         {stk::mesh::InvalidOrdinal};

  bool useShifted_{false};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif
