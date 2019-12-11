// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef MDOTOPENCORRECTORALG_H
#define MDOTOPENCORRECTORALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

namespace sierra {
namespace nalu {

class MdotAlgDriver;

template<typename BcAlgTraits>
class MdotOpenCorrectorAlg : public Algorithm
{
public:
  MdotOpenCorrectorAlg(Realm&, stk::mesh::Part*, MdotAlgDriver&);

  virtual ~MdotOpenCorrectorAlg() = default;

  virtual void execute() override;

private:
  MdotAlgDriver& mdotDriver_;

  const unsigned openMassFlowRate_ {stk::mesh::InvalidOrdinal};
};

}  // nalu
}  // sierra


#endif /* MDOTOPENCORRECTORALG_H */
