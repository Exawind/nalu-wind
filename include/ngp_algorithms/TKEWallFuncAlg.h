/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TKEWALLFUNCALG_H
#define TKEWALLFUNCALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

template<typename BcAlgTraits>
class TKEWallFuncAlg : public Algorithm
{
public:
  TKEWallFuncAlg(Realm&, stk::mesh::Part*);

  virtual ~TKEWallFuncAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests faceData_;

  unsigned bcNodalTke_ {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_  {stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_ {stk::mesh::InvalidOrdinal};

  DoubleType cMu_;

  MasterElement* meFC_{nullptr};
};

}  // nalu
}  // sierra


#endif /* TKEWALLFUNCALG_H */
