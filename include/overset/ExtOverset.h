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

  void set_communicator();

  void breadboard();

  void initialize();

  void update_connectivity();

  void exchange_solution();

  bool multi_solver_mode() const { return multiSolverMode_; }

  void set_multi_solver_mode(const bool flag)
  { multiSolverMode_ = flag; }

  bool is_external_overset() const { return isExtOverset_; }

private:
  TimeIntegrator& time_;

#ifdef NALU_USES_TIOGA
  std::vector<tioga_nalu::TiogaSTKIface*> tgIfaceVec_;
#endif

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
