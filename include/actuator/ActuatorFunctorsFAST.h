// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATORFUNCTORSFAST_H_
#define ACTUATORFUNCTORSFAST_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulkFAST.h>
#include <actuator/ActuatorFunctors.h>
#include <NaluEnv.h>

namespace sierra {
namespace nalu {

struct
ActFastZero{
  using execution_space=ActuatorExecutionSpace;

  ActFastZero(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorMemSpace> helper_;
  ActVectorDbl vel_;
  ActVectorDbl force_;
  ActVectorDbl point_;

};

struct
ActFastUpdatePoints{
  using execution_space=ActuatorFixedExecutionSpace;

  ActFastUpdatePoints(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl points_;
  ActFixScalarInt offsets_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

struct
ActFastAssignVel{
  using execution_space=ActuatorFixedExecutionSpace;

  ActFastAssignVel(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl velocity_;
  ActFixScalarInt offset_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

struct
ActFastComputeForce{
  using execution_space=ActuatorFixedExecutionSpace;

  ActFastComputeForce(ActuatorBulkFAST& actBulk);
  void operator()(int index) const;

  ActDualViewHelper<ActuatorFixedMemSpace> helper_;
  ActFixVectorDbl force_;
  ActFixScalarInt offset_;
  const int turbId_;
  fast::OpenFAST& fast_;
};

struct ActFastSetUpThrustCalc{
  using execution_space = ActuatorFixedExecutionSpace;

  ActFastSetUpThrustCalc(ActuatorBulkFAST& actBulk);

  void operator()(int index) const;

  ActuatorBulkFAST& actBulk_;
};

struct ActFastComputeThrustInnerLoop{

  ActFastComputeThrustInnerLoop(ActuatorBulkFAST& actBulk):actBulk_(actBulk){}
  void operator()(const uint64_t pointId, const double* nodeCoords, double* sourceTerm, const double dualNodalVolume, const double scvIp) const;
  void preloop(){}

  ActuatorBulkFAST& actBulk_;
};

using ActFastComputeThrust = GenericLoopOverCoarseSearchResults<ActuatorBulkFAST, ActFastComputeThrustInnerLoop>;

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORFUNCTORSFAST_H_ */
