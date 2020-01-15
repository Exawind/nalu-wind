// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MDOTOPENEDGEALG_H
#define MDOTOPENEDGEALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class Realm;
class MdotAlgDriver;

template<typename BcAlgTraits>
class MdotOpenEdgeAlg : public Algorithm
{
public:
  MdotOpenEdgeAlg(
    Realm&,
    stk::mesh::Part*,
    MdotAlgDriver&);

  virtual ~MdotOpenEdgeAlg() = default;

  virtual void execute() override;

private:
  MdotAlgDriver& mdotDriver_;

  ElemDataRequests elemData_;
  ElemDataRequests faceData_;

  unsigned coordinates_      {stk::mesh::InvalidOrdinal};
  unsigned velocityRTM_      {stk::mesh::InvalidOrdinal};
  unsigned pressure_         {stk::mesh::InvalidOrdinal};
  unsigned pressureBC_       {stk::mesh::InvalidOrdinal};
  unsigned Gpdx_             {stk::mesh::InvalidOrdinal};
  unsigned density_          {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_   {stk::mesh::InvalidOrdinal};
  unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
  unsigned Udiag_            {stk::mesh::InvalidOrdinal};

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* MDOTOPENEDGEALG_H */
