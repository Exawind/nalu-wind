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

#include <actuator/ActuatorBulkFAST.h>
#include <actuator/ActuatorBulkDiskFAST.h>
#include <actuator/ActuatorFunctorsFAST.h>
#include <actuator/UtilitiesActuator.h>
#include <actuator/ActuatorExecutor.h>

namespace sierra {
namespace nalu {

struct ActuatorLineFastNGP: public ActuatorExecutor
{

  ActuatorLineFastNGP(
    const ActuatorMetaFAST& actMeta,
    ActuatorBulkFAST& actBulk,
    stk::mesh::BulkData& stkBulk);

  virtual ~ActuatorLineFastNGP(){};

  void operator()() final;

  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
  ActDualViewHelper<ActuatorMemSpace> dualViewHelper_;
};

struct ActuatorDiskFastNGP : public ActuatorExecutor
{
  ActuatorDiskFastNGP(
    const ActuatorMetaFAST& actMeta,
    ActuatorBulkDiskFAST& actBulk,
    stk::mesh::BulkData& stkBulk);

  virtual ~ActuatorDiskFastNGP(){};

  void operator()() final;

  const ActuatorMetaFAST& actMeta_;
  ActuatorBulkDiskFAST& actBulk_;
  stk::mesh::BulkData& stkBulk_;
  const int numActPoints_;
  ActDualViewHelper<ActuatorMemSpace> dualViewHelper_;
};

} /* namespace nalu */
} /* namespace sierra */

#endif /* ACTUATORLINEFASTNGP_H_ */
