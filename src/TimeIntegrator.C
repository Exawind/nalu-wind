// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include <TimeIntegrator.h>

#include <Enums.h>
#include <Realm.h>
#include <Realms.h>
#include <Simulation.h>
#include <OutputInfo.h>
#include <SolutionOptions.h>
#include <NaluEnv.h>
#include <NaluParsing.h>
#include <mesh_motion/MeshMotionAlg.h>
#include "overset/ExtOverset.h"

#include <limits>
#include <iomanip>

namespace sierra {
namespace nalu {

TimeIntegrator::TimeIntegrator() {}

TimeIntegrator::TimeIntegrator(Simulation* sim)
  : sim_(sim),
    totalSimTime_(1.0),
    currentTime_(0.0),
    timeStepFromFile_(1.0),
    timeStepN_(1.0),
    timeStepNm1_(1.0),
    gamma1_(1.0),
    gamma2_(-1.0),
    gamma3_(0.0),
    timeStepCount_(0),
    maxTimeStepCount_(std::numeric_limits<int>::max()),
    secondOrderTimeAccurate_(false),
    adaptiveTimeStep_(false),
    terminateBasedOnTime_(false),
    nonlinearIterations_(1),
    overset_(new ExtOverset(*this))
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
TimeIntegrator::~TimeIntegrator()
{
  // does nothing
}

void
TimeIntegrator::load(const YAML::Node& node)
{
  // FIXME - singleton... need TimeIntegrators class...
  const YAML::Node time_integrators = node["Time_Integrators"];
  if (time_integrators) {
    for (size_t itime_int = 0; itime_int < time_integrators.size();
         ++itime_int) {
      const YAML::Node time_int_node = time_integrators[itime_int];
      const YAML::Node standardTimeIntegrator_node =
        time_int_node["StandardTimeIntegrator"];
      // const YAML::Node *otherTimeIntegrator_node =
      // time_int_node.FindValue("OtherTimeIntegrator");
      if (standardTimeIntegrator_node) {
        name_ = standardTimeIntegrator_node["name"].as<std::string>();

        // is termination based on cumulative time or cumulative step count
        if (standardTimeIntegrator_node["termination_time"]) {
          totalSimTime_ =
            standardTimeIntegrator_node["termination_time"].as<double>();
          terminateBasedOnTime_ = true;
        }

        // check for max time step count; will prevail
        if (standardTimeIntegrator_node["termination_step_count"]) {
          maxTimeStepCount_ =
            standardTimeIntegrator_node["termination_step_count"].as<int>();
          if (terminateBasedOnTime_)
            NaluEnv::self().naluOutputP0()
              << "Both max time step and termination time provided, max step "
                 "will prevail"
              << std::endl;
          terminateBasedOnTime_ = false;
        }

        get_if_present(
          standardTimeIntegrator_node, "time_step", timeStepFromFile_,
          timeStepFromFile_);
        get_if_present(
          standardTimeIntegrator_node, "start_time", currentTime_,
          currentTime_);
        get_if_present(
          standardTimeIntegrator_node, "time_step_count", timeStepCount_,
          timeStepCount_);
        get_if_present(
          standardTimeIntegrator_node, "second_order_accuracy",
          secondOrderTimeAccurate_, secondOrderTimeAccurate_);
        get_if_present(
          standardTimeIntegrator_node, "nonlinear_iterations",
          nonlinearIterations_, nonlinearIterations_);

        // set n and nm1 time step; restart will override
        timeStepN_ = timeStepFromFile_;
        timeStepNm1_ = timeStepFromFile_;

        // deal with adaptive dt
        std::string timeStepType = "fixed";
        get_if_present(
          standardTimeIntegrator_node, "time_stepping_type", timeStepType,
          timeStepType);
        adaptiveTimeStep_ = (timeStepType == "fixed") ? false : true;

        NaluEnv::self().naluOutputP0()
          << "StandardTimeIntegrator " << std::endl
          << " name=              " << name_ << std::endl
          << " second order =     " << secondOrderTimeAccurate_ << std::endl;
        if (terminateBasedOnTime_)
          NaluEnv::self().naluOutputP0()
            << " totalSimTime =     " << totalSimTime_ << std::endl;
        else
          NaluEnv::self().naluOutputP0()
            << " maxTimeStepCount = " << maxTimeStepCount_ << std::endl;

        if (adaptiveTimeStep_)
          NaluEnv::self().naluOutputP0()
            << " adaptive time step is active (realm owns specifics) "
            << std::endl;
        else
          NaluEnv::self().naluOutputP0()
            << " fixed time step is active  "
            << " with time step: " << timeStepN_ << std::endl;

        const YAML::Node realms_node = standardTimeIntegrator_node["realms"];
        int iRealm = 0;
        for (size_t irealm = 0; irealm < realms_node.size(); ++irealm) {
          std::string realm_name = realms_node[irealm].as<std::string>();
          NaluEnv::self().naluOutputP0()
            << "StandardTimeIntegrator realm_name[" << iRealm
            << "]= " << realm_name << std::endl;
          realmNamesVec_.push_back(realm_name);
        }
      }
    }
  } else
    throw std::runtime_error("TimeIntegrator::load");
}

void
TimeIntegrator::breadboard()
{
  for (size_t irealm = 0; irealm < realmNamesVec_.size(); ++irealm) {
    Realm* realm = sim_->realms_->find_realm(realmNamesVec_[irealm]);
    realm->timeIntegrator_ = this;
    realmVec_.push_back(realm);
  }

  overset_->breadboard();
}

void
TimeIntegrator::initialize()
{
  overset_->initialize();
  overset_->update_connectivity();
}

void
TimeIntegrator::prepare_for_time_integration()
{
  std::vector<Realm*>::iterator ii;

  //=====================================
  // start-up procedure
  //=====================================

  // initial conditions
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->populate_initial_condition();
  }

  // populate boundary data
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->populate_boundary_data();
  }

  // copy boundary data to solution state
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->boundary_data_to_state_data();
  }

  // read any fields from input file; restoration time returned
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    currentTime_ = (*ii)->populate_variables_from_input(currentTime_);
  }

  // possible restart; need to extract current time (max wins)
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    currentTime_ = std::max(
      currentTime_, (*ii)->populate_restart(timeStepNm1_, timeStepCount_));
  }

  // update connectivity here after all realms have updated their poisitions
  // from possible restarts
  if (!overset_->multi_solver_mode()) {
    overset_->update_connectivity();
  }

  // populate data from transfer; init, io and external
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->process_initialization_transfer();
    // FIXME: might erase the initialization Realm since it has performed its
    // duty (requires shared pointers)
  }

  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->process_io_transfer();
  }

  // read any fields from input file that will serve at external fields
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->populate_external_variables_from_input(currentTime_);
  }
  // process transfer
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->process_external_data_transfer();
  }

  // nm1 dt from possible restart always prevails; input file overrides for
  // fixed time stepping
  if (adaptiveTimeStep_) {
    timeStepN_ = timeStepNm1_;
  } else {
    timeStepN_ = timeStepFromFile_;
  }

  // derived conditions from dofs (interior and boundary)
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->populate_derived_quantities();
  }

  // compute properties based on initial/restart conditions
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->evaluate_properties();
  }

  // perform any initial work
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->initial_work();
  }

  // provide for initial transfer
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->process_multi_physics_transfer();
  }

  if (!overset_->multi_solver_mode()) {
    // provide output/restart for initial condition
    for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
      (*ii)->output_converged_results();
    }
  }
}

void
TimeIntegrator::pre_realm_advance_stage1()
{
  std::vector<Realm*>::iterator ii;

  // negotiate time step
  if (adaptiveTimeStep_) {
    double theStep = 1.0e8;
    for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
      theStep = std::min(theStep, (*ii)->compute_adaptive_time_step());
    }
    timeStepN_ = theStep;
  }

  currentTime_ += timeStepN_;
  timeStepCount_ += 1;

  // compute gamma's
  if (secondOrderTimeAccurate_)
    compute_gamma();

  NaluEnv::self().naluOutputP0()
    << "*******************************************************" << std::endl
    << "Time Step Count: " << timeStepCount_
    << " Current Time: " << currentTime_ << std::endl
    << " dtN: " << timeStepN_ << " dtNm1: " << timeStepNm1_
    << " gammas: " << gamma1_ << " " << gamma2_ << " " << gamma3_ << std::endl;

  // state management
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->swap_states();
    (*ii)->predict_state();
  }

  // read any fields from input file that will serve as external fields
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->populate_external_variables_from_input(currentTime_);
  }

  for (auto realm : realmVec_) {
    realm->update_geometry_due_to_mesh_motion();
  }
}

void
TimeIntegrator::pre_realm_advance_stage2()
{
  std::vector<Realm*>::iterator ii;

  for (auto realm : realmVec_) {
    realm->update_graph_connectivity_and_coordinates_due_to_mesh_motion();
  }

  // populate boundary data
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->populate_boundary_data();
  }

  // output banner
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->output_banner();
  }

  // for this time, extract all of the proper data
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->process_external_data_transfer();
  }
}

void
TimeIntegrator::integrate_realm()
{
  std::vector<Realm*>::iterator ii;

  prepare_for_time_integration();

  bool update_overset = false;
  for (auto* realm : realmVec_) {
    if (realm->does_mesh_move()) {
      update_overset = true;
      break;
    }
  }

  //=====================================
  // time integration
  //=====================================

  while (simulation_proceeds()) {
    const double startTime = NaluEnv::self().nalu_time();

    pre_realm_advance_stage1();
    if (update_overset)
      overset_->update_connectivity();
    pre_realm_advance_stage2();

    const double endPreProc = NaluEnv::self().nalu_time();
    // nonlinear iteration loop; Picard-style
    for (int k = 0; k < nonlinearIterations_; ++k) {
      NaluEnv::self().naluOutputP0()
        << "   Realm Nonlinear Iteration: " << k + 1 << "/"
        << nonlinearIterations_ << std::endl
        << std::endl;

      // update overset and geometry as needed
      interstep_updates(k);

      for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
        (*ii)->advance_time_step();
        (*ii)->process_multi_physics_transfer();
      }
    }

    const double endSolve = NaluEnv::self().nalu_time();
    post_realm_advance();
    const double endPostProc = NaluEnv::self().nalu_time();
    NaluEnv::self().naluOutputP0()
      << "WallClockTime: " << timeStepCount_
      << " Pre: " << (endPreProc - startTime)
      << " NLI: " << (endSolve - endPreProc)
      << " Post: " << (endPostProc - endSolve)
      << " Total: " << (endPostProc - startTime) << std::endl;
  }

  // inform the user that the simulation is complete
  NaluEnv::self().naluOutputP0()
    << "*******************************************************" << std::endl;
  NaluEnv::self().naluOutputP0()
    << "Simulation Shall Complete: time/timestep: " << currentTime_ << "/"
    << timeStepCount_ << std::endl;
  NaluEnv::self().naluOutputP0()
    << "*******************************************************" << std::endl;

  // dump time
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->dump_simulation_time();
  }
}

void
TimeIntegrator::post_realm_advance()
{
  std::vector<Realm*>::iterator ii;

  // process any post converged work
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->post_converged_work();
  }

  // populate data from io transfer
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->process_io_transfer();
  }

  // provide output/restart after nonlinear iteration
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    (*ii)->output_converged_results();
  }

  // output mean norm
  provide_mean_norm();

  timeStepNm1_ = timeStepN_;
}

//--------------------------------------------------------------------------
void
TimeIntegrator::provide_mean_norm()
{
  // provide integrated norm
  std::vector<Realm*>::iterator ii;
  double sumNorm = 0.0;
  double realmIncrement = 0.0;
  for (ii = realmVec_.begin(); ii != realmVec_.end(); ++ii) {
    if ((*ii)->type_ == "multi_physics") {
      // only increment for a "real" realm
      sumNorm += (*ii)->provide_mean_norm();
      realmIncrement += 1.0;
    }
  }
  NaluEnv::self().naluOutputP0()
    << "Mean System Norm: " << std::setprecision(16) << sumNorm / realmIncrement
    << " " << std::setprecision(6) << timeStepCount_ << " " << currentTime_
    << std::endl;
}

//--------------------------------------------------------------------------
void
TimeIntegrator::interstep_updates(int nonLinearIterationIndex)
{
  // hard code as false for now. We want to only trigger this when there is
  // FSI so we need to add an indicator for fsi
  const bool updateGeomInsideNL = nonLinearIterationIndex > 0 && false;

  // perform mesh motion, recompute geometry, etc fsi
  if (updateGeomInsideNL) {
    bool updateOverset = false;

    for (auto&& realm : realmVec_) {
      realm->update_geometry_due_to_mesh_motion();
      if (realm->does_mesh_move()) {
        updateOverset = true;
      }
    }

    if (updateOverset) {
      overset_->update_connectivity();
    }
  }

  if (overset_->is_external_overset()) {
    overset_->exchange_solution();
  }

  // reinit linear systems, reset current coordinates and other topology
  // computations
  if (updateGeomInsideNL) {
    for (auto&& realm : realmVec_) {
      realm->update_graph_connectivity_and_coordinates_due_to_mesh_motion();
    }
  }
}

//--------------------------------------------------------------------------
bool
TimeIntegrator::simulation_proceeds()
{
  bool proceed = false;
  if (terminateBasedOnTime_) {
    if (currentTime_ < totalSimTime_)
      proceed = true;
  } else {
    if (timeStepCount_ < maxTimeStepCount_)
      proceed = true;
  }
  return proceed;
}

//--------------------------------------------------------------------------
double
TimeIntegrator::get_time_step(const NaluState& theState) const
{
  double dt = timeStepN_;
  switch (theState) {
  case NALU_STATE_N:
    dt = timeStepN_;
    break;
  case NALU_STATE_NM1:
    dt = timeStepNm1_;
    break;
  default:
    throw std::runtime_error("unknown state");
  }
  return dt;
}

//--------------------------------------------------------------------------
double
TimeIntegrator::get_current_time() const
{
  return currentTime_;
}

//--------------------------------------------------------------------------
void
TimeIntegrator::compute_gamma()
{
  // defaults
  gamma1_ = 1.0;
  gamma2_ = -1.0;
  gamma3_ = 0.0;

  if (timeStepCount_ > 1) {
    const double tau = timeStepN_ / timeStepNm1_;
    gamma1_ = (1.0 + 2.0 * tau) / (1.0 + tau);
    gamma2_ = -(1.0 + tau);
    gamma3_ = tau * tau / (1.0 + tau);
  }
}

//--------------------------------------------------------------------------
double
TimeIntegrator::get_gamma1() const
{
  return gamma1_;
}

//--------------------------------------------------------------------------
double
TimeIntegrator::get_gamma2() const
{
  return gamma2_;
}

//--------------------------------------------------------------------------
double
TimeIntegrator::get_gamma3() const
{
  return gamma3_;
}

int
TimeIntegrator::get_time_step_count() const
{
  return timeStepCount_;
}

double
TimeIntegrator::get_time_step_from_file()
{
  return timeStepFromFile_;
}

bool
TimeIntegrator::get_is_fixed_time_step()
{
  return !adaptiveTimeStep_;
}

bool
TimeIntegrator::get_is_terminate_based_on_time()
{
  return terminateBasedOnTime_;
}

double
TimeIntegrator::get_total_sim_time()
{
  if (terminateBasedOnTime_)
    return totalSimTime_;
  else
    return -1.0;
}

int
TimeIntegrator::get_max_time_step_count()
{
  if (!terminateBasedOnTime_)
    return maxTimeStepCount_;
  else
    return -1;
}

} // namespace nalu
} // namespace sierra
