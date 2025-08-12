// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef HYPRELINEARSYSTEM_H
#define HYPRELINEARSYSTEM_H

#ifndef HYPRE_LINEAR_SYSTEM_TIMER
#define HYPRE_LINEAR_SYSTEM_TIMER
#endif // HYPRE_LINEAR_SYSTEM_TIMER
#undef HYPRE_LINEAR_SYSTEM_TIMER

#ifndef HYPRE_LINEAR_SYSTEM_DEBUG
#define HYPRE_LINEAR_SYSTEM_DEBUG
#endif // HYPRE_LINEAR_SYSTEM_DEBUG
#undef HYPRE_LINEAR_SYSTEM_DEBUG

#include "KokkosInterface.h"
#include <Kokkos_UnorderedMap.hpp>
#include "LinearSystem.h"
#include "HypreDirectSolver.h"

// This is needed fro get_gpu_memory_info
#include "stk_util/environment/memory_util.hpp"

// NGP Algorithms
#include "ngp_utils/NgpLoopUtils.h"

// These are all necessary for compilation
#include "EquationSystem.h"
#include "NaluEnv.h"
#include "NonConformalManager.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"
#include <utils/CreateDeviceExpression.h>

namespace sierra {
namespace nalu {

using UnsignedView = Kokkos::View<unsigned*, sierra::nalu::MemSpace>;
using UnsignedViewHost = UnsignedView::HostMirror;

using DoubleView = Kokkos::View<double*, sierra::nalu::MemSpace>;
using DoubleViewHost = DoubleView::HostMirror;

using HypreIntTypeView = Kokkos::View<HypreIntType*, sierra::nalu::MemSpace>;
using HypreIntTypeViewHost = HypreIntTypeView::HostMirror;

// const random access views for read only, noncoalesced (texture) memory
// fetches
using UnsignedViewRA = Kokkos::View<
  const unsigned*,
  sierra::nalu::MemSpace,
  Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
using HypreIntTypeViewRA = Kokkos::View<
  const HypreIntType*,
  sierra::nalu::MemSpace,
  Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

// This 2D view needs to be LayoutLeft. Do NOT change
using DoubleView2D =
  Kokkos::View<double**, Kokkos::LayoutLeft, sierra::nalu::MemSpace>;
using DoubleView2DHost = DoubleView2D::HostMirror;

// This 2D view needs to be LayoutLeft. Do NOT change
using HypreIntTypeView2D =
  Kokkos::View<HypreIntType**, Kokkos::LayoutLeft, sierra::nalu::MemSpace>;
using HypreIntTypeView2DHost = HypreIntTypeView2D::HostMirror;

using HypreIntTypeViewScalar =
  Kokkos::View<HypreIntType, sierra::nalu::MemSpace>;
using HypreIntTypeViewScalarHost = HypreIntTypeViewScalar::HostMirror;

using HypreIntTypeUnorderedMap =
  Kokkos::UnorderedMap<HypreIntType, HypreIntType, sierra::nalu::MemSpace>;
using HypreIntTypeUnorderedMapHost = HypreIntTypeUnorderedMap::HostMirror;

using MemoryMap =
  Kokkos::UnorderedMap<HypreIntType, unsigned, sierra::nalu::MemSpace>;
using MemoryMapHost = MemoryMap::HostMirror;

// Periodic Node Map
using PeriodicNodeMap =
  Kokkos::UnorderedMap<HypreIntType, HypreIntType, sierra::nalu::MemSpace>;
using PeriodicNodeMapHost = PeriodicNodeMap::HostMirror;

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
   * @param[in] realm The realm instance that holds the EquationSystem being
   * solved
   * @param[in] numDof The degrees of freedom for the equation system created
   * (Default: 1)
   * @param[in] eqSys The equation system instance
   * @param[in] linearSolver Handle to the HypreDirectSolver instance
   */
  HypreLinearSystem(
    Realm& realm,
    const unsigned numDof,
    EquationSystem* eqSys,
    LinearSolver* linearSolver);

  virtual ~HypreLinearSystem();

  /* equation system name */
  std::string name_;

  /* data structures for accumulating the matrix elements */
  std::vector<HypreIntType> localMatSharedRowCounts_;
  std::vector<HypreIntType> globalMatSharedRowCounts_;
  std::vector<HypreIntType> localRhsSharedRowCounts_;
  std::vector<HypreIntType> globalRhsSharedRowCounts_;
  HypreIntType offProcNNZToSend_;
  HypreIntType offProcNNZToRecv_;
  HypreIntType offProcRhsToSend_;
  HypreIntType offProcRhsToRecv_;

  std::vector<std::vector<HypreIntType>> columnsOwned_;
  std::vector<HypreIntType> rowCountOwned_;

  std::map<HypreIntType, std::vector<HypreIntType>> columnsShared_;
  std::map<HypreIntType, unsigned> rowCountShared_;

  HypreIntTypeViewHost row_indices_owned_host_;
  HypreIntTypeViewHost row_counts_owned_host_;

  HypreIntTypeViewHost row_indices_shared_host_;
  HypreIntTypeViewHost row_counts_shared_host_;

  HypreIntTypeViewHost cols_owned_host_;
  HypreIntTypeViewHost cols_shared_host_;
  HypreIntTypeViewHost cols_host_;

  HypreIntTypeView rows_dev_;
  HypreIntTypeViewHost rows_host_;

  HypreIntTypeView2D rhs_rows_dev_;
  HypreIntTypeView2DHost rhs_rows_host_;

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  FILE* output_ = NULL;
  char oname_[50];
#endif

#ifdef HYPRE_LINEAR_SYSTEM_TIMER
  struct timeval _start, _stop;
  std::vector<double> buildBeginLinSysConstTimer_;
  std::vector<double> buildNodeGraphTimer_;
  std::vector<double> buildFaceToNodeGraphTimer_;
  std::vector<double> buildEdgeToNodeGraphTimer_;
  std::vector<double> buildElemToNodeGraphTimer_;
  std::vector<double> buildFaceElemToNodeGraphTimer_;
  std::vector<double> buildOversetNodeGraphTimer_;
  std::vector<double> buildDirichletNodeGraphTimer_;
  std::vector<double> buildGraphTimer_;
  std::vector<double> finalizeLinearSystemTimer_;
  std::vector<double> hypreMatAssemblyTimer_;
  std::vector<double> hypreRhsAssemblyTimer_;
#endif

  // Quiet "partially overridden" compiler warnings.
  using LinearSystem::buildDirichletNodeGraph;
  // Graph/Matrix Construction
  virtual void buildNodeGraph(
    const stk::mesh::PartVector&
      parts); // for nodal assembly (e.g., lumped mass and source)
  virtual void buildFaceToNodeGraph(
    const stk::mesh::PartVector& parts); // face->node assembly
  virtual void buildEdgeToNodeGraph(
    const stk::mesh::PartVector& parts); // edge->node assembly
  virtual void buildElemToNodeGraph(
    const stk::mesh::PartVector& parts); // elem->node assembly
  virtual void buildReducedElemToNodeGraph(
    const stk::mesh::PartVector&); // elem (nearest nodes only)->node assembly
  virtual void buildFaceElemToNodeGraph(
    const stk::mesh::PartVector& parts); // elem:face->node assembly
  virtual void buildNonConformalNodeGraph(
    const stk::mesh::PartVector&); // nonConformal->elem_node assembly
  virtual void buildOversetNodeGraph(
    const stk::mesh::PartVector&); // overset->elem_node assembly
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
  virtual void
  buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes);

  /** Finalize construction of the linear system matrix and rhs vector
   *
   *  This method calls the appropriate Hypre functions to assemble the matrix
   *  and rhs in a parallel run, as well as registers the matrix and rhs with
   *  the solver preconditioner.
   */
  virtual void loadComplete();

  virtual void dumpMatrixStats();

  /** Reset the matrix and rhs data structures for the next iteration/timestep
   *
   */
  virtual void zeroSystem();

  /** Solve the system Ax = b
   *
   *  The solution vector is returned in linearSolutionField
   *
   *  @param[out] linearSolutionField STK field where the solution is populated
   */
  virtual int solve(stk::mesh::FieldBase* linearSolutionField);

  //! Helper method to transfer the solution from a HYPRE_IJVector instance to
  //! the STK field data instance.
  double copy_hypre_to_stk(stk::mesh::FieldBase*);

#ifdef HYPRE_LINEAR_SYSTEM_DEBUG
  void scanBufferForBadValues(
    double* ptr,
    int N,
    const char* file,
    const char* func,
    int line,
    char* bufferName);
  void scanOwnedIndicesForBadValues(
    HypreIntType* rows,
    HypreIntType* cols,
    int N,
    const char* file,
    const char* func,
    int line);
  void scanSharedIndicesForBadValues(
    HypreIntType* rows,
    HypreIntType* cols,
    int N,
    const char* file,
    const char* func,
    int line);
#endif

  /** Populate the LHS and RHS for the Dirichlet rows in linear system
   */
  virtual void applyDirichletBCs(
    stk::mesh::FieldBase* solutionField,
    stk::mesh::FieldBase* bcValuesField,
    const stk::mesh::PartVector& parts,
    const unsigned beginPos,
    const unsigned endPos);

  sierra::nalu::CoeffApplier* get_coeff_applier();

  // print timings for initialize
  virtual void printTimings(std::vector<double>& time, const char* name);
  virtual void buildCoeffApplierPeriodicNodeToHIDMapping();
  virtual void resetCoeffApplierData();
  virtual void finishCoupledOversetAssembly();
  virtual void hypreIJMatrixSetAddToValues();
  virtual void hypreIJVectorSetAddToValues();
  virtual void buildCoeffApplierDeviceOwnedDataStructures();
  virtual void buildCoeffApplierDeviceSharedDataStructures();
  virtual void buildCoeffApplierDeviceDataStructures();
  virtual void computeRowSizes();
  virtual void fill_hids_columns(
    const unsigned numNodes,
    stk::mesh::Entity const* nodes,
    std::vector<HypreIntType>& hids,
    std::vector<HypreIntType>& columns);
  virtual void fill_owned_shared_data_structures_1DoF(
    const unsigned numNodes, std::vector<HypreIntType>& hids);
  virtual void fill_owned_shared_data_structures(
    const unsigned numNodes,
    std::vector<HypreIntType>& hids,
    std::vector<HypreIntType>& columns);

  /***************************************************************************************************/
  /*                     Beginning of HypreLinSysCoeffApplier definition */
  /***************************************************************************************************/

  class HypreLinSysCoeffApplier : public CoeffApplier
  {
  public:
    HypreLinSysCoeffApplier(
      unsigned numDof, unsigned nDim, HypreIntType iLower, HypreIntType iUpper);

    KOKKOS_DEFAULTED_FUNCTION
    virtual ~HypreLinSysCoeffApplier() = default;

    KOKKOS_FUNCTION
    virtual void reset_rows(
      unsigned numNodes,
      const stk::mesh::Entity* nodeList,
      const double diag_value,
      const double rhs_residual,
      const HypreIntType iLower,
      const HypreIntType iUpper,
      const unsigned numDof,
      HypreIntType memShift);

    KOKKOS_FUNCTION
    virtual void resetRows(
      unsigned numNodes,
      const stk::mesh::Entity* nodeList,
      const unsigned,
      const unsigned,
      const double diag_value,
      const double rhs_residual);

    KOKKOS_FUNCTION
    virtual void sort(
      const SharedMemView<int*, DeviceShmem>& localIds,
      const SharedMemView<int*, DeviceShmem>& sortPermutation,
      unsigned N);

    KOKKOS_FUNCTION
    virtual void sum_into(
      unsigned numEntities,
      const stk::mesh::NgpMesh::ConnectedNodes& entities,
      const SharedMemView<int*, DeviceShmem>& localIds,
      const SharedMemView<int*, DeviceShmem>& sortPermutation,
      const SharedMemView<const double*, DeviceShmem>& rhs,
      const SharedMemView<const double**, DeviceShmem>& lhs,
      const HypreIntType& iLower,
      const HypreIntType& iUpper,
      unsigned numDof,
      HypreIntType memShift);

    KOKKOS_FUNCTION
    virtual void sum_into_1DoF(
      unsigned numEntities,
      const stk::mesh::NgpMesh::ConnectedNodes& entities,
      const SharedMemView<int*, DeviceShmem>& localIds,
      const SharedMemView<int*, DeviceShmem>& sortPermutation,
      const SharedMemView<const double*, DeviceShmem>& rhs,
      const SharedMemView<const double**, DeviceShmem>& lhs,
      const HypreIntType& iLower,
      const HypreIntType& iUpper,
      HypreIntType memShift);

    KOKKOS_FUNCTION
    virtual void operator()(
      unsigned numEntities,
      const stk::mesh::NgpMesh::ConnectedNodes& entities,
      const SharedMemView<int*, DeviceShmem>& localIds,
      const SharedMemView<int*, DeviceShmem>& sortPermutation,
      const SharedMemView<const double*, DeviceShmem>& rhs,
      const SharedMemView<const double**, DeviceShmem>& lhs,
      const char* trace_tag);

    virtual void free_device_pointer();

    virtual sierra::nalu::CoeffApplier* device_pointer();

    //! mesh
    stk::mesh::NgpMesh ngpMesh_;
    //! stk mesh field for the Hypre Global Id
    NGPHypreIDFieldType ngpHypreGlobalId_;
    //! number of degrees of freedom
    unsigned numDof_ = 0;
    //! number of rhs vectors
    unsigned nDim_ = 0;
    //! The lowest row owned by this MPI rank
    HypreIntType iLower_ = 0;
    //! The highest row owned by this MPI rank
    HypreIntType iUpper_ = 0;

    /* monolithic data structures for holding all the values for
       the owned and shared parts. Shared MUST come after owned. */
    DoubleView values_dev_;
    HypreIntTypeView cols_dev_;
    DoubleView2D rhs_dev_;

    //! Data structures for the owned CSR Matrix and RHS Vector(s)
    HypreIntType num_rows_owned_;
    HypreIntType num_nonzeros_owned_;
    UnsignedView mat_row_start_owned_;
    HypreIntTypeView periodic_bc_rows_owned_;

    //! Data structures for the shared CSR Matrix and RHS Vector(s)
    HypreIntType num_rows_shared_;
    HypreIntType num_nonzeros_shared_;
    MemoryMap map_shared_;
    UnsignedView mat_row_start_shared_;
    UnsignedView rhs_row_start_shared_;

    //! Random access views
    UnsignedViewRA mat_row_start_owned_ra_;
    UnsignedViewRA mat_row_start_shared_ra_;
    HypreIntTypeViewRA cols_dev_ra_;

    //! Auxilliary Data structures

    //! map of the periodic nodes to hypre ids
    PeriodicNodeMap periodic_node_to_hypre_id_;

    //! Flag indicating that sumInto should check to see if rows must be skipped
    HypreIntTypeViewScalar checkSkippedRows_;
    //! unordered map for skipped rows
    HypreIntTypeUnorderedMap skippedRowsMap_;
    HypreIntTypeUnorderedMapHost skippedRowsMapHost_;

    //! unordered map for overset rows
    HypreIntTypeUnorderedMap oversetRowsMap_;
    HypreIntTypeUnorderedMapHost oversetRowsMapHost_;

    //! this is the pointer to the device function ... that assembles the lists
    HypreLinSysCoeffApplier* devicePointer_;

    /* flag to reinitialize or not */
    bool reinitialize_ = true;

    //! number of points in the overset data structures
    HypreIntType num_mat_overset_pts_owned_;
    HypreIntType num_rhs_overset_pts_owned_;

    /* Work space for overset. These are used to accumulate data from legacy,
     * non-NGP sumInto calls */
    HypreIntTypeView d_overset_row_indices_;
    HypreIntTypeViewHost h_overset_row_indices_;

    HypreIntTypeView d_overset_rows_;
    HypreIntTypeView d_overset_cols_;
    HypreIntTypeViewHost h_overset_rows_;
    HypreIntTypeViewHost h_overset_cols_;

    DoubleView d_overset_vals_;
    DoubleViewHost h_overset_vals_;

    DoubleView d_overset_rhs_vals_;
    DoubleViewHost h_overset_rhs_vals_;

    /* counters for adding to the array */
    int overset_mat_counter_ = 0;
    int overset_rhs_counter_ = 0;
  };

  /***************************************************************************************************/
  /*                        End of of HypreLinSysCoeffApplier definition */
  /***************************************************************************************************/

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
    const std::vector<stk::mesh::Entity>& sym_meshobj,
    std::vector<int>& scratchIds,
    std::vector<double>& scratchVals,
    const std::vector<double>& rhs,
    const std::vector<double>& lhs,
    const char* trace_tag);

  /*****************************************/
  /* Legacy methods needed for Compilation */
  /*****************************************/
  virtual void sumInto(
    unsigned /*numEntities*/,
    const stk::mesh::NgpMesh::ConnectedNodes& /*entities*/,
    const SharedMemView<const double*, DeviceShmem>& /*rhs*/,
    const SharedMemView<const double**, DeviceShmem>& /*lhs*/,
    const SharedMemView<int*, DeviceShmem>& /*localIds*/,
    const SharedMemView<int*, DeviceShmem>& /*sortPermutation*/,
    const char* /*trace_tag*/)
  {
  }

  virtual void resetRows(
    const std::vector<stk::mesh::Entity>&,
    const unsigned,
    const unsigned,
    const double,
    const double)
  {
  }

  virtual void resetRows(
    unsigned /*numNodes*/,
    const stk::mesh::Entity* /*nodeList*/,
    const unsigned,
    const unsigned,
    const double,
    const double)
  {
  }

  virtual void
  writeToFile(const char* /* filename */, bool /* useOwned */ = true)
  {
  }
  virtual void
  writeSolutionToFile(const char* /* filename */, bool /* useOwned */ = true)
  {
  }

protected:
  /** Prepare the instance for system construction
   *
   *  During initialization, this creates the hypre data structures via API
   *  calls. It also synchronizes hypreGlobalId across shared and ghosted data
   *  so that hypre row ID lookups succeed during initialization and assembly.
   */
  virtual void beginLinearSystemConstruction();

  virtual void loadCompleteSolver();

  /** Return the Hypre ID corresponding to the given STK node entity
   *
   *  @param[in] entity The STK node entity object
   *
   *  @return The HYPRE row ID
   */
  HypreIntType get_entity_hypre_id(const stk::mesh::Entity&);

  /** Dummy method to satisfy inheritance
   */
  void checkError(const int, const char*) {}

  //! The HYPRE matrix data structure
  mutable HYPRE_IJMatrix mat_;

  //! Track which rows are skipped
  std::unordered_set<HypreIntType> skippedRows_;

  //! Track which rows are skipped
  std::unordered_set<HypreIntType> oversetRows_;

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
  //! Total number of rows owned by this particular MPI rank
  HypreIntType globalNumRows_;
  //! Store the rank as class data so it's easy to reference
  int rank_;
  //! Maximum Row ID in the Hypre linear system
  HypreIntType maxRowID_;

  //! Flag indicating whether IJMatrixAssemble has been called on the system
  bool hypreMatrixVectorsCreated_{false};

  //! Flag indicating whether the linear system has been initialized
  bool matrixStatsDumped_{false};

private:
  //! HYPRE right hand side data structure
  mutable HYPRE_IJVector rhs_;

  //! HYPRE solution vector
  mutable HYPRE_IJVector sln_;
};

} // namespace nalu
} // namespace sierra

#endif /* HYPRELINEARSYSTEM_H */
