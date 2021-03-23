// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MDOTINFLOWALG_H
#define MDOTINFLOWALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class Realm;
class MdotAlgDriver;

template <typename BcAlgTraits>
class MdotInflowAlg : public Algorithm
{
public:
  MdotInflowAlg(Realm&, stk::mesh::Part*, MdotAlgDriver&, bool);

  virtual ~MdotInflowAlg() = default;

  virtual void execute() override;

private:
  MdotAlgDriver& mdotDriver_;

  ElemDataRequests faceData_;

  unsigned velocityBC_{stk::mesh::InvalidOrdinal};
  unsigned density_{stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_{stk::mesh::InvalidOrdinal};
  unsigned edgeFaceVelMag_{stk::mesh::InvalidOrdinal};

  bool useShifted_;

  MasterElement* meFC_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* MDOTINFLOWALG_H */
