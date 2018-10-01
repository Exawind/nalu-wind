/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef HYPREDIRECTSOLVER_H
#define HYPREDIRECTSOLVER_H

#include "LinearSolver.h"

#include "XSDKHypreInterface.h"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

namespace sierra {
namespace nalu {

/** Nalu interface to Hypre Solvers and Preconditioners
 *
 *  This class is responsible creation, initialization, execution, and clean up
 *  of Hypre solver and preconditioner data structures during the simulation. It
 *  provides an abstraction layer so that the user can choose different Hypre
 *  solvers via input parameters. This class interacts with rest of Nalu solely
 *  via sierra::nalu::HypreLinearSystem. The configuration of Hypre solver is
 *  controlled via user input parameters processed in
 *  sierra::nalu::HypreLinearSolverConfig
 *
 *  Users are referred to the [Hypre Reference
 *  Manual](https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software)
 *  for detailed documentation on the Hypre functions and data structures used
 *  in this class.
 */
class HypreDirectSolver: public LinearSolver
{
public:
  HypreDirectSolver(
    std::string name,
    HypreLinearSolverConfig* config,
    LinearSolvers* linearSolvers);

  virtual ~HypreDirectSolver();

  //! Clean up Hypre data structures during simulation
  virtual void destroyLinearSolver() override;

  /** Solves the linear system and updates the solution vector.
   *
   *  @param iters The number of linear iterations performed
   *  @param norm The norm of the final relative residual
   */
  int solve(int&, double&, bool);

  //! Return the type of solver instance
  virtual PetraType getType() override { return PT_HYPRE; }

  //! Instance of the Hypre parallel matrix
  mutable HYPRE_ParCSRMatrix parMat_;

  //! Instance of the Hypre parallel RHS vector
  mutable HYPRE_ParVector parRhs_;

  //! Instance of Hypre parallel solution vector
  mutable HYPRE_ParVector parSln_;

  MPI_Comm comm_;

protected:
  //! Helper method to handle processing the user inputs and creating the
  //! appropriate solver/preconditioner instances.
  void initSolver();

  virtual void setupSolver();

  //! Create the Hypre Solver and related call methods
  void createSolver();

  //! Create the Hypre preconditioner and related call methods
  void createPrecond();

  //! Enum indicating the solver type used in this simulation
  Ifpack2::Hypre::Hypre_Solver solverType_;

  //! Enum indicating the preconditioner type used in this simulation
  Ifpack2::Hypre::Hypre_Solver precondType_;

  mutable HYPRE_Solver solver_;

  mutable HYPRE_Solver precond_;

  HypreIntType (*solverCreatePtr_)(MPI_Comm, HYPRE_Solver*);
  HypreIntType (*solverDestroyPtr_)(HYPRE_Solver);
  HypreIntType (*solverSetupPtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  HypreIntType (*solverSolvePtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  HypreIntType (*solverPrecondPtr_)(
    HYPRE_Solver,
    HYPRE_PtrToParSolverFcn,
    HYPRE_PtrToParSolverFcn,
    HYPRE_Solver);

  HypreIntType (*precondCreatePtr_)(MPI_Comm, HYPRE_Solver*);
  HypreIntType (*precondDestroyPtr_)(HYPRE_Solver);
  HypreIntType (*precondSetupPtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
  HypreIntType (*precondSolvePtr_)(
    HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);

  HypreIntType (*solverSetTolPtr_)(HYPRE_Solver, double);
  HypreIntType (*solverNumItersPtr_)(HYPRE_Solver, HypreIntType*);
  HypreIntType (*solverFinalResidualNormPtr_)(HYPRE_Solver, double*);


  //! Flag indicating whether a preconditioner is used. Certain solvers like
  //! BoomerAMG do not require a preconditioner.
  bool usePrecond_{false};

  //! Flag indicating whether the Hypre solver has been setup. This is used to
  //! determine whether the Destroy functions are called to clean up
  bool isSolverSetup_{false};

  //! Flag indicating whether the Hypre preconditioner has been setup. This is
  //! used to determine whether the Destroy functions are called to clean up
  bool isPrecondSetup_{false};

  //! Flag indicating whether this class instance has been initialized fully
  bool isInitialized_{false};

private:
  HypreDirectSolver() = delete;
  HypreDirectSolver(const HypreDirectSolver&) = delete;
};

}  // nalu
}  // sierra

#endif /* HYPREDIRECTSOLVER_H */
