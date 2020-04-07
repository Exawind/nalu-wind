// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFUNCTORS_H_
#define ACTUATORFUNCTORS_H_

#include <actuator/ActuatorGenericSearchFunctor.h>
#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulk.h>
#include <FieldTypeDef.h>

namespace stk{
namespace mesh{
  class BulkData;
}
}

namespace sierra
{
namespace nalu
{

struct InterpActuatorVel{
  using execution_space = ActuatorFixedExecutionSpace;

  InterpActuatorVel(ActuatorBulk& actBulk, stk::mesh::BulkData& stkBulk);

  void operator()(int index) const;

  ActuatorBulk& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  VectorFieldType* coordinates_;
  VectorFieldType* velocity_;
};



struct SpreadForceInnerLoop{
  SpreadForceInnerLoop(ActuatorBulk& actBulk):actBulk_(actBulk){}

  void operator()(const uint64_t pointId, const double* nodeCoords, double* sourceTerm, const double dualNodalVolume, const double scvIp) const;
  void preloop();

  ActuatorBulk& actBulk_;
};

using SpreadActuatorForce = GenericLoopOverCoarseSearchResults<ActuatorBulk, SpreadForceInnerLoop>;

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORS_H_ */
