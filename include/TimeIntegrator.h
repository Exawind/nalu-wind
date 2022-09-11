// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef TimeIntegrator_h
#define TimeIntegrator_h

#include <Enums.h>
#include <vector>
#include <string>
#include <memory>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

class Realm;
class Simulation;
class ExtOverset;

class TimeIntegrator
{
public:
  TimeIntegrator();
  TimeIntegrator(Simulation* sim);
  ~TimeIntegrator();

  void load(const YAML::Node& node);

  void breadboard();

  void initialize();

  void integrate_realm();
  void provide_mean_norm();
  bool simulation_proceeds();

  void prepare_for_time_integration();
  void pre_realm_advance_stage1();
  void pre_realm_advance_stage2();
  void post_realm_advance();
  void interstep_updates(int nonLinearIterationIndex);

  Simulation* sim_{nullptr};

  double totalSimTime_;
  double currentTime_;
  double timeStepFromFile_;
  double timeStepN_;
  double timeStepNm1_;
  double gamma1_;
  double gamma2_;
  double gamma3_;
  int timeStepCount_;
  int maxTimeStepCount_;
  bool secondOrderTimeAccurate_;
  bool adaptiveTimeStep_;
  bool terminateBasedOnTime_;
  int nonlinearIterations_;

  std::string name_;

  std::vector<std::string> realmNamesVec_;

  std::vector<Realm*> realmVec_;

  double get_time_step(const NaluState& theState = NALU_STATE_N) const;
  double get_current_time() const;
  double get_gamma1() const;
  double get_gamma2() const;
  double get_gamma3() const;
  int get_time_step_count() const;
  double get_time_step_from_file();
  bool get_is_fixed_time_step();
  bool get_is_terminate_based_on_time();
  double get_total_sim_time();
  int get_max_time_step_count();
  void compute_gamma();

  std::unique_ptr<ExtOverset> overset_;
};

} // namespace nalu
} // namespace sierra

#endif
