// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <actuator/ActuatorParsing.h>
#include <actuator/ActuatorModel.h>

#ifdef NALU_USES_OPENFAST
#include <actuator/ActuatorParsingFAST.h>
#include <actuator/ActuatorBulkFAST.h>
#include <actuator/ActuatorExecutorsFASTNgp.h>
#endif

#include <actuator/ActuatorParsingSimple.h>
#include <actuator/ActuatorBulkSimple.h>
#include <actuator/ActuatorExecutorsSimpleNgp.h>
#include <string>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

void
ActuatorModel::parse(const YAML::Node& actuatorNode)
{
  ActuatorMeta actMetaBase = actuator_parse(actuatorNode);
  switch (actMetaBase.actuatorType_) {
  case ActuatorType::ActDiskFASTNGP:
  case ActuatorType::ActLineFASTNGP: {
#ifdef NALU_USES_OPENFAST
    actMeta_.reset(
      new ActuatorMetaFAST(actuator_FAST_parse(actuatorNode, actMetaBase)));
    break;
#else
    throw std::runtime_error(
      "look_ahead_and_create::error: Requested actuator type: " +
      ActuatorTypeName + ", but was not enabled at compile time");
#endif
#ifndef __CUDACC__
    break;
#endif
  }
  case ActuatorType::ActLineSimpleNGP: {
    actMeta_.reset(
      new ActuatorMetaSimple(actuator_Simple_parse(actuatorNode, actMetaBase)));
    break;
  }
  case ActuatorType::ActLineSimple:
  case ActuatorType::ActLineFAST:
  case ActuatorType::ActDiskFAST:
    break; // TODO move these to the appropriate version above when old code is
           // deleted
  default: {
    const std::string actuatorType =
      actuatorNode["actuator"]["type"].as<std::string>();
    throw std::runtime_error(
      "look_ahead_and_create::error: unrecognized actuator type: " +
      actuatorType);
// Avoid nvcc unreachable statement warnings
#ifndef __CUDACC__
    break;
#endif
  }
  }
}

void
ActuatorModel::setup(double timeStep, stk::mesh::BulkData& stkBulk)
{
  if (actMeta_ == NULL)
    return;

  switch (actMeta_->actuatorType_) {
  case (ActuatorType::ActLineFASTNGP): {
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#else
    auto tempMeta = dynamic_cast<ActuatorMetaFAST*>(actMeta_.get());
    auto tempBulk = dynamic_cast<ActuatorBulkFAST*>(actBulk_.get());
    actBulk_.reset(new ActuatorBulkFAST(*tempMeta, timeStep));
    actExec_.reset(new ActuatorLineFastNGP(*tempMeta, *tempBulk, stkBulk));
#endif
#ifndef __CUDACC__
    break;
#endif
  }
  case (ActuatorType::ActDiskFASTNGP): {
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#else
    auto tempMeta = dynamic_cast<ActuatorMetaFAST*>(actMeta_.get());
    auto tempBulk = dynamic_cast<ActuatorBulkDiskFAST*>(actBulk_.get());
    actBulk_.reset(new ActuatorBulkDiskFAST(*tempMeta, timeStep));
    actExec_.reset(new ActuatorDiskFastNGP(*tempMeta, *tempBulk, stkBulk));
#endif
#ifndef __CUDACC__
    break;
#endif
  }
  case (ActuatorType::ActLineSimpleNGP): {
    auto tempMeta = dynamic_cast<ActuatorMetaSimple*>(actMeta_.get());
    auto tempBulk = dynamic_cast<ActuatorBulkSimple*>(actBulk_.get());
    actBulk_.reset(new ActuatorBulkSimple(*tempMeta));
    actExec_.reset(new ActuatorLineSimpleNGP(*tempMeta, *tempBulk, stkBulk));
  }
  default: {
    ThrowErrorMsg("Unsupported actuator type");
  }
  }
}

void
ActuatorModel::init(stk::mesh::BulkData& stkBulk)
{
  if (is_active())
    return;

  switch (actMeta_->actuatorType_) {
  case (ActuatorType::ActLineFASTNGP):
  case (ActuatorType::ActDiskFASTNGP):
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#endif
    // perform search for actline and actdisk
    actBulk_->stk_search_act_pnts(*actMeta_.get(), stkBulk);
    break;
  case (ActuatorType::ActLineSimpleNGP):
  case (ActuatorType::ActLineSimple):
  case (ActuatorType::ActLineFAST):
  case (ActuatorType::ActDiskFAST):
    break;
  default: {
    ThrowErrorMsg("Unsupported actuator type");
  }
  }
}

void
ActuatorModel::execute(double& timer)
{
  if (actExec_ == NULL)
    return;

  const double start_time = NaluEnv::self().nalu_time();
  actExec_->operator()();
  const double end_time = NaluEnv::self().nalu_time();
  timer += end_time - start_time;
}

} // namespace nalu
} // namespace sierra