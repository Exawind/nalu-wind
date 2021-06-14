// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MOMENTUMEDGEPECLETKERNEL_H_
#define MOMENTUMEDGEPECLETKERNEL_H_

#include <Algorithm.h>
#include "stk_mesh/base/Types.hpp"
#include "PecletFunction.h"

namespace sierra{
namespace nalu{

class Realm;
class EquationSystem;

class MomentumEdgePecletAlg: public Algorithm{
public:
  using DblType = double;

  MomentumEdgePecletAlg(Realm& , stk::mesh::Part*, EquationSystem*);
  virtual ~MomentumEdgePecletAlg() = default;
  void execute() override;

  
private:
  unsigned pecletNumber_{stk::mesh::InvalidOrdinal};
  unsigned pecletFactor_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned viscosity_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned vrtm_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  const double eps_{1.0e-16};
  const int nDim_;
  PecletFunction<DblType>* pecletFunction_{nullptr};
};

void determine_max_peclet_factor(stk::mesh::BulkData& bulk, const stk::mesh::MetaData& meta);

void determine_max_peclet_number(stk::mesh::BulkData& bulk, const stk::mesh::MetaData& meta);

}
}


#endif /* MOMENTUMEDGEPECLETKERNEL_H_ */
