// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MDOTDENSITYACCUMALG_H
#define MDOTDENSITYACCUMALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

namespace sierra {
namespace nalu {

class MdotAlgDriver;

template<typename AlgTraits>
class MdotDensityAccumAlg : public Algorithm
{
public:
  MdotDensityAccumAlg(Realm&, stk::mesh::Part*, MdotAlgDriver&, bool);

  virtual ~MdotDensityAccumAlg() = default;

  virtual void execute() override;

private:
  MdotAlgDriver& mdotDriver_;

  ElemDataRequests elemData_;

  const unsigned rhoNp1_ {stk::mesh::InvalidOrdinal};
  const unsigned rhoN_ {stk::mesh::InvalidOrdinal};
  const unsigned rhoNm1_ {stk::mesh::InvalidOrdinal};

  MasterElement* meSCV_{nullptr};

  const bool lumpedMass_{false};
};

}  // nalu
}  // sierra


#endif /* MDOTDENSITYACCUMALG_H */
