// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <aero/actuator/ActuatorParsing.h>
#include <aero/actuator/ActuatorModel.h>

#ifdef NALU_USES_OPENFAST
#include <aero/actuator/ActuatorParsingFAST.h>
#include <aero/actuator/ActuatorBulkFAST.h>
#include <aero/actuator/ActuatorExecutorsFASTNgp.h>
#endif

#include <aero/actuator/ActuatorParsingSimple.h>
#include <aero/actuator/ActuatorBulkSimple.h>
#include <aero/actuator/ActuatorExecutorsSimpleNgp.h>
#include <string>
#include <NaluParsing.h>

namespace sierra {
namespace nalu {

void
ActuatorModel::parse(const YAML::Node& actuatorNode)
{
  const ActuatorMeta actMetaBase = actuator_parse(actuatorNode);
  const std::string actuatorType =
    actuatorNode["actuator"]["type"].as<std::string>();
  switch (actMetaBase.actuatorType_) {
  case ActuatorType::ActDiskFASTNGP:
  case ActuatorType::ActLineFASTNGP: {
#ifdef NALU_USES_OPENFAST
    actMeta_.reset(
      new ActuatorMetaFAST(actuator_FAST_parse(actuatorNode, actMetaBase)));
    break;
#else
    throw std::runtime_error(
      "look_ahead_and_create::error: Requested actuator type: " + actuatorType +
      ", but was not enabled at compile time");
#endif
#ifndef KOKKOS_ENABLE_CUDA
    break;
#endif
  }
  case ActuatorType::ActLineSimpleNGP: {
    actMeta_.reset(
      new ActuatorMetaSimple(actuator_Simple_parse(actuatorNode, actMetaBase)));
    break;
  }
  default: {
    throw std::runtime_error(
      "look_ahead_and_create::error: unrecognized actuator type: " +
      actuatorType);
// Avoid nvcc unreachable statement warnings
#ifndef KOKKOS_ENABLE_CUDA
    break;
#endif
  }
  }
}

void
ActuatorModel::setup(double timeStep, stk::mesh::BulkData& stkBulk)
{
  if (!is_active())
    return;

  // hack to surpress Wunused-parameter on non-openfast builds
  ThrowErrorIf(timeStep <= 0.0);

  switch (actMeta_->actuatorType_) {
  case (ActuatorType::ActLineFASTNGP): {
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#ifndef KOKKOS_ENABLE_CUDA
    break;
#endif
#else
    auto tempMeta =
      dcast::dcast_and_check_pointer<ActuatorMeta, ActuatorMetaFAST>(
        actMeta_.get());
    actBulk_.reset(new ActuatorBulkFAST(*tempMeta, timeStep));
    auto tempBulk =
      dcast::dcast_and_check_pointer<ActuatorBulk, ActuatorBulkFAST>(
        actBulk_.get());
    actExec_.reset(new ActuatorLineFastNGP(*tempMeta, *tempBulk, stkBulk));
    break;
#endif
  }
  case (ActuatorType::ActDiskFASTNGP): {
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#ifndef KOKKOS_ENABLE_CUDA
    break;
#endif
#else
    auto tempMeta =
      dcast::dcast_and_check_pointer<ActuatorMeta, ActuatorMetaFAST>(
        actMeta_.get());
    actBulk_.reset(new ActuatorBulkDiskFAST(*tempMeta, timeStep));
    auto tempBulk =
      dcast::dcast_and_check_pointer<ActuatorBulk, ActuatorBulkDiskFAST>(
        actBulk_.get());
    actExec_.reset(new ActuatorDiskFastNGP(*tempMeta, *tempBulk, stkBulk));
    break;
#endif
  }
  case (ActuatorType::ActLineSimpleNGP): {
    auto tempMeta =
      dcast::dcast_and_check_pointer<ActuatorMeta, ActuatorMetaSimple>(
        actMeta_.get());
    actBulk_.reset(new ActuatorBulkSimple(*tempMeta));
    auto tempBulk =
      dcast::dcast_and_check_pointer<ActuatorBulk, ActuatorBulkSimple>(
        actBulk_.get());
    actExec_.reset(new ActuatorLineSimpleNGP(*tempMeta, *tempBulk, stkBulk));
    break;
  }
  default: {
    ThrowErrorMsg("Unsupported actuator type");
  }
  }
}

void
ActuatorModel::init(stk::mesh::BulkData& stkBulk)
{
  if (!is_active())
    return;

  // do nothing call to surpress warning when built w/o openfast
  (void)stkBulk;

  switch (actMeta_->actuatorType_) {
  case (ActuatorType::ActLineFASTNGP):
  case (ActuatorType::ActDiskFASTNGP): {
#ifndef NALU_USES_OPENFAST
    ThrowErrorMsg("Actuator methods require OpenFAST");
#ifndef KOKKOS_ENABLE_CUDA
    break;
#endif
#else
    // perform search for actline and actdisk
    actBulk_->stk_search_act_pnts(*actMeta_.get(), stkBulk);
    break;
#endif
  }
  case (ActuatorType::ActLineSimpleNGP): {
    break;
  }
  default: {
    ThrowErrorMsg("Unsupported actuator type");
  }
  }
}

void
ActuatorModel::execute(double& timer)
{
  if (!is_active())
    return;

  const double start_time = NaluEnv::self().nalu_time();
  actExec_->operator()();
  const double end_time = NaluEnv::self().nalu_time();
  timer += end_time - start_time;
}

} // namespace nalu
} // namespace sierra
