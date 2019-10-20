/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef SDRLOWREWALLFUNCALG_H
#define SDRLOWREWALLFUNCALG_H

#include "Algorithm.h"
#include "ElemDataRequests.h"
#include "SimdInterface.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra {
namespace nalu {

/** Specific Dissipation Rate low-Reynolds number wall boundary value
 *
 *. \sa SDRWallFuncAlgDriver
 */
template<typename BcAlgTraits>
class SDRLowReWallAlg : public Algorithm
{
public:
  SDRLowReWallAlg(
    Realm&,
    stk::mesh::Part*,
    const bool);

  virtual ~SDRLowReWallAlg() = default;

  virtual void execute() override;

private:
  ElemDataRequests faceData_;
  ElemDataRequests elemData_;

  unsigned coordinates_    {stk::mesh::InvalidOrdinal};
  unsigned density_        {stk::mesh::InvalidOrdinal};
  unsigned viscosity_      {stk::mesh::InvalidOrdinal};
  unsigned exposedAreaVec_ {stk::mesh::InvalidOrdinal};
  unsigned wallArea_       {stk::mesh::InvalidOrdinal};
  unsigned sdrbc_          {stk::mesh::InvalidOrdinal};

  const DoubleType betaOne_;
  const DoubleType wallFactor_;
  const bool useShifted_;

  MasterElement* meFC_{nullptr};
  MasterElement* meSCS_{nullptr};
};

}  // nalu
}  // sierra


#endif /* SDRLOWREWALLFUNCALG_H */
