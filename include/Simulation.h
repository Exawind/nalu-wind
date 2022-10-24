// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef Simulation_h
#define Simulation_h

#include <stk_util/diag/PrintTimer.hpp>
#include <stk_util/diag/Timer.hpp>

#include <KokkosInterface.h>

namespace YAML {
class Node;
}

namespace sierra {
namespace nalu {

class LinearSolvers;
class TimeIntegrator;
class Realms;
class Transfers;

class Simulation
{
public:
  Simulation(const YAML::Node& root_node);

  ~Simulation();

  void load(const YAML::Node& node);
  void breadboard();
  void initialize();
  void init_prolog();
  void init_epilog();
  void run();
  void high_level_banner();
  Simulation* root() { return this; }
  Simulation* parent() { return 0; }
  bool debug() { return debug_; }
  bool debug() const { return debug_; }
  void setSerializedIOGroupSize(int siogs);
  static stk::diag::TimerSet& rootTimerSet();
  static stk::diag::Timer& rootTimer();
  static stk::diag::Timer& outputTimer();

  const YAML::Node& m_root_node;
  TimeIntegrator* timeIntegrator_;
  Realms* realms_;
  Transfers* transfers_;
  LinearSolvers* linearSolvers_;

  static bool debug_;
  int serializedIOGroupSize_;

private:
#if defined(KOKKOS_ENABLE_GPU)
  size_t default_stack_size;
  const size_t nalu_stack_size = 16384;
#endif
};

} // namespace nalu
} // namespace sierra

#endif
