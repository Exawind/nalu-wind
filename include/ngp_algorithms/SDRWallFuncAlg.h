/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
    stk::mesh::Part*);

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
};

}  // nalu
}  // sierra


#endif /* SDRWALLFUNCALG_H */
