// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef STRELETSUPWINDEDGEALG_H_
#define STRELETSUPWINDEDGEALG_H_

#include <string>
#include <Algorithm.h>
#include "stk_mesh/base/Types.hpp"

namespace sierra{
namespace nalu{

class Realm;

class StreletsUpwindEdgeAlg : public Algorithm{
public:
  using DblType = double;
  StreletsUpwindEdgeAlg(Realm&, stk::mesh::Part*);
  virtual ~StreletsUpwindEdgeAlg() = default;
  void execute() override;

private:
  const std::string velocityName_;
  unsigned pecletFactor_ {stk::mesh::InvalidOrdinal};
  unsigned fOne_ {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolume_ {stk::mesh::InvalidOrdinal};
  unsigned sstMaxLen_ {stk::mesh::InvalidOrdinal};
  unsigned dudx_ {stk::mesh::InvalidOrdinal};
  unsigned density_ {stk::mesh::InvalidOrdinal};
  unsigned viscosity_ {stk::mesh::InvalidOrdinal};
  unsigned turbViscosity_ {stk::mesh::InvalidOrdinal};
  unsigned turbKE_ {stk::mesh::InvalidOrdinal};
  unsigned specDissRate_ {stk::mesh::InvalidOrdinal};
  unsigned edgeAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned coordinates_ {stk::mesh::InvalidOrdinal};
  unsigned velocity_ {stk::mesh::InvalidOrdinal};
};

}
}


#endif /* STRELETSUPWINDEDGEALG_H_ */
