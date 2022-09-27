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

#include <aero/actuator/ActuatorBulkSimple.h>
#include <aero/actuator/ActuatorFunctorsSimple.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <aero/actuator/ActuatorExecutor.h>
#include <memory>

namespace sierra {
namespace nalu {

class ActuatorLineSimpleNGP : public ActuatorExecutor
{
public:
  ActuatorLineSimpleNGP(
    const ActuatorMetaSimple& actMeta,
    ActuatorBulkSimple& actBulk,
    stk::mesh::BulkData& stkBulk);

  virtual ~ActuatorLineSimpleNGP(){};
  void operator()() final;

private:
  const ActuatorMetaSimple& actMeta_;
  ActuatorBulkSimple& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
  const bool useSpreadActuatorForce_;
};

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINESIMPLENGP_H_ */
