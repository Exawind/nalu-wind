// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef EXTOVERSET_H
#define EXTOVERSET_H

#include "TimeIntegrator.h"
#include "stk_util/parallel/Parallel.hpp"

namespace tioga_nalu {
class TiogaSTKIface;
}

namespace sierra {
namespace nalu {

class Realm;

class ExtOverset
{
public:
  ExtOverset(TimeIntegrator& timeIntegrator_);

  ~ExtOverset();

  //! Initialize MPI comm for TIOGA instance
  void set_communicator();

  //! Set up mesh metadata structures within overset instances
  void breadboard();

  /** Perform initial connectivity between participating meshes
   */
  void initialize();

  /** Update overset connectivity for moving meshes
   */
  void update_connectivity();

  //! Update solution fields using TIOGA
  void exchange_solution();

  //! Register meshes to TIOGA to perform overset connectivity with external meshes
  void pre_overset_conn_work();

  //! Perform IBLANK updates (and ghosting if necessary) after connectivity
  void post_overset_conn_work();

  //! Register solution fields to TIOGA before interpolation step
  int register_solution(const std::vector<std::string>& fnames);

  //! Update solution fields after TIOGA has performed interpolations
  void update_solution();

  bool multi_solver_mode() const { return multiSolverMode_; }

  void set_multi_solver_mode(const bool flag)
  { multiSolverMode_ = flag; }

  bool is_external_overset() const { return isExtOverset_; }

private:
  TimeIntegrator& time_;

#ifdef NALU_USES_TIOGA
  std::vector<tioga_nalu::TiogaSTKIface*> tgIfaceVec_;
#endif

  std::vector<std::string> slnFieldNames_;

  //! Flag indicating whether we are interfacing external solver
  bool multiSolverMode_{false};

  //! Flag indicating whether there are multiple realms
  bool isExtOverset_{false};

  //! Is the algorithm decoupled solves
  bool isDecoupled_{true};

  //! Is there any type of overset algorithm available
  bool hasOverset_{false};
};

}  // nalu
}  // sierra


#endif /* EXTOVERSET_H */
