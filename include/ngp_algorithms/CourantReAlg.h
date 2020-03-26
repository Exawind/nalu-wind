// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef COURANTREALG_H
#define COURANTREALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

class Realm;
class CourantReAlgDriver;

template<typename AlgTraits>
class CourantReAlg : public Algorithm
{
public:
  CourantReAlg(Realm&, stk::mesh::Part*, CourantReAlgDriver&);

  virtual ~CourantReAlg() = default;

  virtual void execute() override;

private:
  CourantReAlgDriver& algDriver_;

  ElemDataRequests elemData_;

  const unsigned coordinates_{stk::mesh::InvalidOrdinal};
  const unsigned velocity_{stk::mesh::InvalidOrdinal};
  const unsigned density_{stk::mesh::InvalidOrdinal};
  const unsigned viscosity_{stk::mesh::InvalidOrdinal};
  const unsigned elemCFL_{stk::mesh::InvalidOrdinal};
  const unsigned elemRe_{stk::mesh::InvalidOrdinal};

  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* COURANTREALG_H */
