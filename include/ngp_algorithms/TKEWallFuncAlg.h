// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef TKEWALLFUNCALG_H
#define TKEWALLFUNCALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

/** Compute nodal TKE values when using ABL wall function
 *
 *  Use a driver/algorithm style design to allow the actual computation to be
 *  templated on the face element type. Computation of the "wall_model_tke_bc"
 *  field is performed in TKEWallFuncAlg class. This field is synchronized in
 *  parallel as well as host/device in TKEWallFuncAlgDriver::post_work and
 *  "turbulent_ke" and "tke_bc" fields are updated for the wall boundaries where
 *  wall function is being used.
 *
 *  \sa TKEWallFuncAlgDriver
 */
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
