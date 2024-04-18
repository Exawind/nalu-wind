// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "overset/ExtOverset.h"
#include "overset/TiogaRef.h"
#include "overset/OversetManagerTIOGA.h"
#include "overset/UpdateOversetFringeAlgorithmDriver.h"
#include "overset/overset_utils.h"
#include "NaluEnv.h"
#include "Realm.h"

#ifdef NALU_USES_TIOGA
#include "tioga.h"
#endif

namespace sierra {
namespace nalu {

ExtOverset::ExtOverset(TimeIntegrator& time) : time_(time) {}

ExtOverset::~ExtOverset() = default;

void
ExtOverset::set_communicator()
{
#ifdef NALU_USES_TIOGA
  auto& tg = tioga_nalu::TiogaRef::self().get();

  auto& env = NaluEnv::self();
  tg.setCommunicator(
    env.parallel_comm(), env.parallel_rank(), env.parallel_size());
#endif
}

void
ExtOverset::breadboard()
{
  int noverset = 0;
  for (auto* realm : time_.realmVec_) {
    if (realm->query_for_overset()) {
      ++noverset;

      isDecoupled_ =
        isDecoupled_ && realm->equationSystems_.all_systems_decoupled();
    }
  }

  if (multiSolverMode_ && (noverset == 0)) {
    throw std::runtime_error(
      "Multi-solver mode is active but no realm has overset");
  }

  if (!isDecoupled_ && (noverset > 1)) {
    throw std::runtime_error("External overset requires decoupled solves");
  }

  if (noverset > 0)
    hasOverset_ = true;
  if (noverset > 1 || multiSolverMode_) {
    for (auto* realm : time_.realmVec_)
      realm->isExternalOverset_ = true;
  }

  isExtOverset_ = multiSolverMode_ || (noverset > 1);

  if (!multiSolverMode_ && hasOverset_)
    set_communicator();
}

void
ExtOverset::initialize()
{
  if (!hasOverset_)
    return;

#ifdef NALU_USES_TIOGA
  for (auto* realm : time_.realmVec_) {
    if (!realm->hasOverset_)
      continue;

    auto* mgr = dynamic_cast<OversetManagerTIOGA*>(realm->oversetManager_);
    tgIfaceVec_.push_back(&mgr->tiogaIface_);

    mgr->initialize();
  }
#endif
}

void
ExtOverset::update_connectivity()
{
  if (!hasOverset_)
    return;

#ifdef NALU_USES_TIOGA
  auto& tg = tioga_nalu::TiogaRef::self().get();

  for (auto* tgiface : tgIfaceVec_) {
    tgiface->register_mesh();
  }

  tg.profile();
  tg.performConnectivity();

  for (auto* tgiface : tgIfaceVec_) {
    tgiface->post_connectivity_work(isDecoupled_);
  }
#endif
}

void
ExtOverset::pre_overset_conn_work()
{
  if (!hasOverset_)
    return;

#ifdef NALU_USES_TIOGA
  for (auto* tgiface : tgIfaceVec_) {
    tgiface->register_mesh();
  }
#endif
}

void
ExtOverset::post_overset_conn_work()
{
  if (!hasOverset_)
    return;

#ifdef NALU_USES_TIOGA
  for (auto* tgiface : tgIfaceVec_) {
    tgiface->post_connectivity_work(isDecoupled_);
  }
#endif
}

void
ExtOverset::exchange_solution()
{
  if (!hasOverset_)
    return;

#ifdef NALU_USES_TIOGA
  const int row_major = 0;
  auto& tg = tioga_nalu::TiogaRef::self().get();

  int ncomp = 0;
  for (auto* realm : time_.realmVec_) {
    if (!realm->hasOverset_)
      continue;

    auto& mgr =
      dynamic_cast<OversetManagerTIOGA*>(realm->oversetManager_)->tiogaIface_;
    ncomp =
      mgr.register_solution(realm->equationSystems_.oversetUpdater_->fields_);
  }

  tg.dataUpdate(ncomp, row_major);

  for (auto* realm : time_.realmVec_) {
    if (!realm->hasOverset_)
      continue;

    auto& mgr =
      dynamic_cast<OversetManagerTIOGA*>(realm->oversetManager_)->tiogaIface_;
    mgr.update_solution(realm->equationSystems_.oversetUpdater_->fields_);
  }
#endif
}

int
ExtOverset::register_solution(const std::vector<std::string>& fnames)
{
  int ncomp = 0;
  if (!hasOverset_)
    return ncomp;

  STK_ThrowAssert(fnames.size() > 0u);
  // Store field names for update solution phase
  slnFieldNames_ = fnames;

#ifdef NALU_USES_TIOGA
  for (auto* realm : time_.realmVec_) {
    if (!realm->hasOverset_)
      continue;

    const auto fields =
      overset_utils::get_overset_field_data(*realm, slnFieldNames_);
    auto& mgr =
      dynamic_cast<OversetManagerTIOGA*>(realm->oversetManager_)->tiogaIface_;
    ncomp = mgr.register_solution(fields);
  }
#endif
  return ncomp;
}

void
ExtOverset::update_solution()
{
  if (!hasOverset_)
    return;

  STK_ThrowAssert(slnFieldNames_.size() > 0u);

#ifdef NALU_USES_TIOGA
  for (auto* realm : time_.realmVec_) {
    if (!realm->hasOverset_)
      continue;

    const auto fields =
      overset_utils::get_overset_field_data(*realm, slnFieldNames_);
    auto& mgr =
      dynamic_cast<OversetManagerTIOGA*>(realm->oversetManager_)->tiogaIface_;
    mgr.update_solution(fields);
  }
#endif
}

} // namespace nalu
} // namespace sierra
