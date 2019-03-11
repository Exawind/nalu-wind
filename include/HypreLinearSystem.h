/*------------------------------------------------------------------------*/
/*  Copyright 2014 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef HYPRELINEARSYSTEM_H
#define HYPRELINEARSYSTEM_H

#include "LinearSystem.h"
#include "XSDKHypreInterface.h"

#include "stk_mesh/base/BulkData.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"

#include <unordered_set>

namespace sierra {
namespace nalu {

/** Nalu interface to populate a Hypre Linear System
 *
 *  This class provides an interface to the HYPRE IJMatrix and IJVector data
 *  structures. It is responsible for creating, resetting, and destroying the
 *  Hypre data structures and provides the HypreLinearSystem::sumInto interface
 *  used by Nalu Kernels and SupplementalAlgorithms to populate entries into the
 *  linear system. The HypreLinearSystem::solve method interfaces with
 *  sierra::nalu::HypreDirectSolver that is responsible for the actual solution
 *  of the system using the required solver and preconditioner combination.
 */
class HypreLinearSystem : public LinearSystem
{
public:
  /**
   * @param[in] realm The realm instance that holds the EquationSystem being solved
   * @param[in] numDof The degrees of freedom for the equation system created (Default: 1)
   * @param[in] eqSys The equation system instance
   * @param[in] linearSolver Handle to the HypreDirectSolver instance
   */
  HypreLinearSystem(
    Realm& realm,
    const unsigned numDof,
    EquationSystem *eqSys,
    LinearSolver *linearSolver);

  virtual ~HypreLinearSystem();

  // Graph/Matrix Construction
  virtual void buildNodeGraph(const stk::mesh::PartVector&);// for nodal assembly (e.g., lumped mass and source)
  virtual void buildFaceToNodeGraph(const stk::mesh::PartVector&);// face->node assembly
  virtual void buildEdgeToNodeGraph(const stk::mesh::PartVector&);// edge->node assembly
  virtual void buildElemToNodeGraph(const stk::mesh::PartVector&);// elem->node assembly
  virtual void buildReducedElemToNodeGraph(const stk::mesh::PartVector&);// elem (nearest nodes only)->node assembly
  virtual void buildFaceElemToNodeGraph(const stk::mesh::PartVector&);// elem:face->node assembly
  virtual void buildNonConformalNodeGraph(const stk::mesh::PartVector&);// nonConformal->elem_node assembly
  virtual void buildOversetNodeGraph(const stk::mesh::PartVector&);// overset->elem_node assembly
  virtual void finalizeLinearSystem();

  /** Tag rows that must be handled as a Dirichlet BC node
   *
   *  @param[in] partVec List of parts that contain the Dirichlet nodes
   */
  virtual void buildDirichletNodeGraph(const stk::mesh::PartVector&);

  /** Tag rows that must be handled as a Dirichlet  node
   *
   *  @param[in] entities List of nodes where Dirichlet conditions are applied
   *
   *  \sa sierra::nalu::FixPressureAtNodeAlgorithm
   */
  virtual void buildDirichletNodeGraph(const std::vector<stk::mesh::Entity>&);

  /** Reset the matrix and rhs data structures for the next iteration/timestep
   *
   */
  virtual void zeroSystem();

  /** Update coefficients of a particular row(s) in the linear system
   *
   *  The core method of this class, it updates the matrix and RHS based on the
   *  inputs from the various algorithms. Note that, unlike TpetraLinearSystem,
   *  this method skips over the fringe points of Overset mesh and the Dirichlet
   *  nodes rather than resetting them afterward.
   *
   *  This overloaded method deals with Kernels designed with Kokkos::View arrays.
   *
   *  @param[in] numEntities The total number of nodes where data is to be updated
   *  @param[in] entities A list of STK node entities
   *
   *  @param[in] rhs Array containing RHS entries to be summed into
   *      [numEntities * numDof]
   *
   *  @param[in] lhs Array containing LHS entries to be summed into.
   *      [numEntities * numDof, numEntities * numDof]
   *
   *  @param[in] localIds Work array for storing local row IDs
   *  @param[in] sortPermutation Work array for sorting row IDs
   *  @param[in] trace_tag Debugging message
   */
  virtual void sumInto(
      unsigned numEntities,
      const stk::mesh::Entity* entities,
      const SharedMemView<const double*> & rhs,
      const SharedMemView<const double**> & lhs,
      const SharedMemView<int*> & localIds,
      const SharedMemView<int*> & sortPermutation,
      const char * trace_tag
  );

  /** Update coefficients of a particular row(s) in the linear system
   *
   *  The core method of this class, it updates the matrix and RHS based on the
   *  inputs from the various algorithms. Note that, unlike TpetraLinearSystem,
   *  this method skips over the fringe points of Overset mesh and the Dirichlet
   *  nodes rather than resetting them afterward.
   *
   *  This overloaded method deals with classic SupplementalAlgorithms
   *
   *  @param[in] sym_meshobj A list of STK node entities
   *  @param[in] scratchIds Work array for row IDs
   *  @param[in] scratchVals Work array for row entries
   *
   *  @param[in] rhs Array containing RHS entries to be summed into
   *      [numEntities * numDof]
   *
   *  @param[in] lhs Array containing LHS entries to be summed into.
   *      [numEntities * numDof * numEntities * numDof]
   *
   *  @param[in] trace_tag Debugging message
   */
  virtual void sumInto(
    const std::vector<stk::mesh::Entity> & sym_meshobj,
    std::vector<int> &scratchIds,
    std::vector<double> &scratchVals,
    const std::vector<double> & rhs,
    const std::vector<double> & lhs,
    const char *trace_tag
  );

  /** Populate the LHS and RHS for the Dirichlet rows in linear system
   */
  virtual void applyDirichletBCs(
    stk::mesh::FieldBase * solutionField,
    stk::mesh::FieldBase * bcValuesField,
    const stk::mesh::PartVector & parts,
    const unsigned beginPos,
    const unsigned endPos);

  /** Prepare assembly for overset fringe nodes
   *
   *  The overset fringe nodes are skipped over by the sumInto method during
   *  normal assembly process. This method toggles the flag to instruct sumInto
   *  that the constraint rows are being filled at this stage.
   */
  virtual void prepareConstraints(
    const unsigned,
    const unsigned)
  {
    checkSkippedRows_ = false;
  }

  /** Prepare assembly for Dirichlet-type rows
   *
   *  Dirichlet rows are skipped over by the sumInto method when the interior
   *  parts are processed. This method toggles the flag alerting the sumInto
   *  method that the Dirichlet rows will be processed next and sumInto can
   *  proceed.
   */
  virtual void resetRows(
    const std::vector<stk::mesh::Entity>&,
    const unsigned,
    const unsigned,
    const double,
    const double)
  {
    checkSkippedRows_ = false;
  }

  /** Solve the system Ax = b
   *
   *  The solution vector is returned in linearSolutionField
   *
   *  @param[out] linearSolutionField STK field where the solution is populated
   */
  virtual int solve(stk::mesh::FieldBase * linearSolutionField);

  /** Finalize construction of the linear system matrix and rhs vector
   *
   *  This method calls the appropriate Hypre functions to assemble the matrix
   *  and rhs in a parallel run, as well as registers the matrix and rhs with
   *  the solver preconditioner.
   */
  virtual void loadComplete();

  virtual void writeToFile(const char * /* filename */, bool /* useOwned */ =true) {}
  virtual void writeSolutionToFile(const char * /* filename */, bool /* useOwned */ =true) {}

protected:
  /** Prepare the instance for system construction
   *
   *  During initialization, this creates the hypre data structures via API
   *  calls. It also synchronizes hypreGlobalId across shared and ghosted data
   *  so that hypre row ID lookups succeed during initialization and assembly.
   */
  virtual void beginLinearSystemConstruction();

  virtual void finalizeSolver();

  virtual void loadCompleteSolver();

  /** Return the Hypre ID corresponding to the given STK node entity
   *
   *  @param[in] entity The STK node entity object
   *
   *  @return The HYPRE row ID
   */
  HypreIntType get_entity_hypre_id(const stk::mesh::Entity&);

  //! Helper method to transfer the solution from a HYPRE_IJVector instance to
  //! the STK field data instance.
  double copy_hypre_to_stk(stk::mesh::FieldBase*);

  /** Flags indicating whether a particular row in the HYPRE matrix has been
   * filled or not.
   */
  enum RowFillStatus
  {
    RS_UNFILLED = 0, //!< Default status
    RS_FILLED        //!< sumInto filps to filled status once a row has been acted on
  };

  /** Flag indicating the type of row.
   *
   *  This flag is used to determine if the normal sumInto approach is used to
   *  populate the row, or a special method is used to handle that row. sumInto
   *  method will skip over the rows not marked RT_NORMAL and must be dealt with
   *  separately by other algorithms.
   */
  enum RowStatus
  {
    RT_NORMAL = 0, //!< A normal row that is summed into using sumInto
    RT_DIRICHLET,  //!< Rows with Dirichlet BC; no off-diagonal entries
    RT_OVERSET     //!< Overset fringe points; interpolation weights from other mesh
  };

  /** Dummy method to satisfy inheritance
   */
  void checkError(
    const int,
    const char*) {}

  //! The HYPRE matrix data structure
  mutable HYPRE_IJMatrix mat_;

  //! Track rows that have been updated during the assembly process
  std::vector<RowFillStatus> rowFilled_;

  //! Track the status of rows
  std::vector<RowStatus> rowStatus_;

  //! Track which rows are skipped
  std::unordered_set<HypreIntType> skippedRows_;

  //! Buffer for handling Global Row IDs for use in sumInto methods
  std::vector<HypreIntType> idBuffer_;

  //! The lowest row owned by this MPI rank
  HypreIntType iLower_;
  //! The highest row owned by this MPI rank
  HypreIntType iUpper_;
  //! The lowest column owned by this MPI rank; currently jLower_ == iLower_
  HypreIntType jLower_;
  //! The highest column owned by this MPI rank; currently jUpper_ == iUpper_
  HypreIntType jUpper_;
  //! Total number of rows owned by this particular MPI rank
  HypreIntType numRows_;
  //! Maximum Row ID in the Hypre linear system
  HypreIntType maxRowID_;

  //! Flag indicating whether IJMatrixAssemble has been called on the system
  bool matrixAssembled_{false};

  //! Flag indicating whether the linear system has been initialized
  bool systemInitialized_{false};

  //! Flag indicating that sumInto should check to see if rows must be skipped
  bool checkSkippedRows_{false};

  //! Flag indicating that dirichlet and/or overset rows are present for this system
  bool hasSkippedRows_{false};

private:
  //! HYPRE right hand side data structure
  mutable HYPRE_IJVector rhs_;

  //! HYPRE solution vector
  mutable HYPRE_IJVector sln_;

};

}  // nalu
}  // sierra


#endif /* HYPRELINEARSYSTEM_H */
