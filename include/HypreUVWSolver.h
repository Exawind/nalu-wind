/*------------------------------------------------------------------------*/
/*  Copyright 2018 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
