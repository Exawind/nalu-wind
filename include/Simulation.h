/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef Simulation_h
#define Simulation_h

#include <stk_util/diag/PrintTimer.hpp>
#include <stk_util/diag/Timer.hpp>

namespace YAML {
class Node;
}

namespace sierra{
namespace nalu{

class LinearSolvers;
class TimeIntegrator;
class Realms;
class Transfers;

class Simulation {
public:
  Simulation(const YAML::Node& root_node);

  ~Simulation();

  void load(const YAML::Node & node);
  void breadboard();
  void initialize();
  void run();
  void high_level_banner();
  Simulation *root() { return this; }
  Simulation *parent() { return 0; }
  bool debug() { return debug_; }
  bool debug() const { return debug_; }
  void setSerializedIOGroupSize(int siogs);
  static stk::diag::TimerSet &rootTimerSet();
  static stk::diag::Timer &rootTimer();
  static stk::diag::Timer &outputTimer();

  const YAML::Node& m_root_node;
  TimeIntegrator *timeIntegrator_;
  Realms *realms_;
  Transfers * transfers_;
  LinearSolvers *linearSolvers_;

  static bool debug_;
  int serializedIOGroupSize_;
};

} // namespace nalu
} // namespace Sierra

#endif

