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

namespace stk{ namespace mesh{ class BulkData; } }

namespace sierra{
namespace nalu{

class Realm;
class EquationSystem;

class MomentumEdgePecletAlg: public Algorithm{
public:
  // use simd due to ngp_peclet function
  using DblType = double;

  MomentumEdgePecletAlg(Realm& , stk::mesh::Part*, EquationSystem*);
  virtual ~MomentumEdgePecletAlg() = default;
  void execute() override;

  
private:
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

}
}


#endif /* MOMENTUMEDGEPECLETKERNEL_H_ */
