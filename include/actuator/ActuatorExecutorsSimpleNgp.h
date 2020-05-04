// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATOREXECUTORSSIMPLENGP_H_
#define ACTUATOREXECUTORSSIMPLENGP_H_

#include <actuator/ActuatorNGP.h>
#include <actuator/ActuatorBulkSimple.h>
//#include <actuator/ActuatorBulkDiskSimple.h>
#include <actuator/ActuatorFunctorsSimple.h>
#include <actuator/UtilitiesActuator.h>

namespace sierra {
namespace nalu {

struct ActuatorLineSimpleNGP
{

  ActuatorLineSimpleNGP(
    const ActuatorMetaSimple& actMeta,
    ActuatorBulkSimple& actBulk,
    stk::mesh::BulkData& stkBulk);

  void operator()();

  void update();

  const ActuatorMetaSimple& actMeta_;
  ActuatorBulkSimple& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
  ActDualViewHelper<ActuatorMemSpace> dualViewHelper_;
};
/*
struct ActuatorDiskFastNGP
{
  ActuatorDiskFastNGP(
    const ActuatorMetaFAST& actMeta,
    ActuatorBulkDiskFAST& actBulk,
    stk::mesh::BulkData& stkBulk);

  void operator()();

  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkDiskFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
  ActDualViewHelper<ActuatorMemSpace> dualViewHelper_;
};
*/
} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINESIMPLENGP_H_ */
