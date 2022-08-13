// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ACTUATOREXECUTORSFASTNGP_H_
#define ACTUATOREXECUTORSFASTNGP_H_

#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/ActuatorBulkDiskFAST.h>
#include <aero/actuator/ActuatorFunctorsFAST.h>
#include <aero/actuator/UtilitiesActuator.h>
#include <aero/actuator/ActuatorExecutor.h>

namespace sierra {
namespace nalu {

class ActuatorLineFastNGP : public ActuatorExecutor
{
public:
  ActuatorLineFastNGP(
    const ActuatorMetaFAST& actMeta,
    ActuatorBulkFAST& actBulk,
    stk::mesh::BulkData& stkBulk);

  virtual ~ActuatorLineFastNGP(){};

  void operator()() final;

private:
  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
};

class ActuatorDiskFastNGP : public ActuatorExecutor
{
public:
  ActuatorDiskFastNGP(
    const ActuatorMetaFAST& actMeta,
    ActuatorBulkDiskFAST& actBulk,
    stk::mesh::BulkData& stkBulk);

  virtual ~ActuatorDiskFastNGP(){};

  void operator()() final;

private:
  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkDiskFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
};

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINEFASTNGP_H_ */
