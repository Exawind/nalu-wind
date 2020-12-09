// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef SDRWALLFUNCALG_H
#define SDRWALLFUNCALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

/** SDR Wall function using wall friction velocity (u_tau)
 *
 *  \sa SDRWallFuncAlgDriver
 */
template<typename BcAlgTraits>
class SDRWallFuncAlg : public Algorithm
{
public:
  SDRWallFuncAlg(
    Realm&,
    stk::mesh::Part*,
    bool = false,
    double = 0.0);

  virtual ~SDRWallFuncAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests faceData_;
  ElemDataRequests elemData_;

  unsigned coordinates_    {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned wallFricVel_    {stk::mesh::InvalidOrdinal};
  unsigned wallArea_       {stk::mesh::InvalidOrdinal};
  unsigned sdrbc_          {stk::mesh::InvalidOrdinal};

  const DoubleType sqrtBetaStar_;
  const DoubleType kappa_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};

  bool RANSAblBcApproach_;
  double z0_;
};

}  // nalu
}  // sierra


#endif /* SDRWALLFUNCALG_H */
