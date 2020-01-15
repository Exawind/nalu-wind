// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef HYPREUVWSOLVER_H
#define HYPREUVWSOLVER_H

#include "HypreDirectSolver.h"

namespace sierra {
namespace nalu {

class HypreUVWSolver: public HypreDirectSolver
{
public:
  HypreUVWSolver(
    std::string,
    HypreLinearSolverConfig*,
    LinearSolvers*);

  virtual ~HypreUVWSolver();

  int solve(int, int&, double&, bool);

  //! Return the type of solver instance
  virtual PetraType getType() { return PT_HYPRE_SEGREGATED; }

  mutable std::vector<HYPRE_ParVector> parRhsU_;

  mutable std::vector<HYPRE_ParVector> parSlnU_;

protected:
  virtual void setupSolver();

private:
  HypreUVWSolver() = delete;
  HypreUVWSolver(const HypreUVWSolver&) = delete;
};

}  // nalu
}  // sierra


#endif /* HYPREUVWSOLVER_H */
