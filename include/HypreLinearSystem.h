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

#include "LinearSystem.h"
#include "XSDKHypreInterface.h"

#include "stk_mesh/base/BulkData.hpp"
#include "stk_mesh/base/NgpMesh.hpp"
#include "stk_mesh/base/MetaData.hpp"
#include "stk_mesh/base/Field.hpp"
#include "stk_mesh/base/FieldParallel.hpp"
#include "stk_util/parallel/ParallelReduce.hpp"

#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "krylov.h"
#include "HYPRE.h"
// why are these included 
#include "_hypre_parcsr_mv.h"
#include "_hypre_IJ_mv.h"
#include "HYPRE_config.h"

#include "HypreDirectSolver.h"
#include "Realm.h"
#include "EquationSystem.h"
#include "LinearSolver.h"
#include "PeriodicManager.h"
#include "NaluEnv.h"
#include "NonConformalManager.h"
#include "overset/OversetManager.h"
#include "overset/OversetInfo.h"

#include <utils/StkHelpers.h>
#include <utils/CreateDeviceExpression.h>

// NGP Algorithms
#include "ngp_utils/NgpLoopUtils.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Vector.hpp>
#include <Kokkos_Sort.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<typename TYPE>
struct absAscendingOrdering
{
  absAscendingOrdering() {}
  __host__ __device__
  bool operator()(const thrust::tuple<TYPE,double>& x, const thrust::tuple<TYPE,double>& y) const
  { 
    TYPE x1 = thrust::get<0>(x);
    double x2 = thrust::get<1>(x);
    TYPE y1 = thrust::get<0>(y);
    double y2 = thrust::get<1>(y);
    if (x1<y1) return true;
    else if (x1>y1) return false;
    else {
      if (abs(x2)<abs(y2)) return true;
      else return false;
    }
  }
};
#endif

namespace sierra {
namespace nalu {

using UnsignedView = Kokkos::View<unsigned*, sierra::nalu::MemSpace>;
using UnsignedViewHost = UnsignedView::HostMirror;

using DoubleView = Kokkos::View<double*, sierra::nalu::MemSpace>;
using DoubleViewHost = DoubleView::HostMirror;

using HypreIntTypeView = Kokkos::View<HypreIntType*, sierra::nalu::MemSpace>;
using HypreIntTypeViewHost = HypreIntTypeView::HostMirror;

// This 2D view needs to be LayoutLeft. Do NOT change
using DoubleView2D = Kokkos::View<double**, Kokkos::LayoutLeft, sierra::nalu::MemSpace>;
using DoubleView2DHost = DoubleView2D::HostMirror;

// This 2D view needs to be LayoutLeft. Do NOT change
using HypreIntTypeView2D = Kokkos::View<HypreIntType**, Kokkos::LayoutLeft, sierra::nalu::MemSpace>;
using HypreIntTypeView2DHost = HypreIntTypeView2D::HostMirror;

using HypreIntTypeViewScalar = Kokkos::View<HypreIntType, sierra::nalu::MemSpace>;
using HypreIntTypeViewScalarHost = HypreIntTypeViewScalar::HostMirror;

using HypreIntTypeUnorderedMap = Kokkos::UnorderedMap<HypreIntType, HypreIntType, sierra::nalu::MemSpace>;
using HypreIntTypeUnorderedMapHost = HypreIntTypeUnorderedMap::HostMirror;

using MemoryMap = Kokkos::UnorderedMap<HypreIntType, unsigned, sierra::nalu::MemSpace>;
using MemoryMapHost = MemoryMap::HostMirror;

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
  std::string name_;

  /* data structures for accumulating the matrix elements */
  std::vector<std::vector<HypreIntType> > columnsOwned_;
  std::vector<HypreIntType> rowCountOwned_;

  std::map<HypreIntType, std::vector<HypreIntType> > columnsShared_;
  std::map<HypreIntType, unsigned> rowCountShared_;

  float _hypreAssembleTime=0.f;
  int _nHypreAssembles=0;

  HypreIntTypeView entityToLID_;
  HypreIntTypeViewHost entityToLIDHost_;

  HypreIntTypeView row_indices_owned_;
  HypreIntTypeView row_counts_owned_;
  HypreIntTypeView periodic_bc_rows_owned_;
  HypreIntTypeView mat_elem_keys_owned_;
  UnsignedView mat_elem_start_owned_;
  UnsignedView mat_row_start_owned_;
  UnsignedView rhs_row_start_owned_;

  MemoryMap mat_map_shared_;
  MemoryMap rhs_map_shared_;
  HypreIntTypeView row_indices_shared_;
  HypreIntTypeView row_counts_shared_;
  UnsignedView mat_elem_start_shared_;
  UnsignedView mat_row_start_shared_;
  UnsignedView rhs_row_start_shared_;

  HypreIntTypeUnorderedMap skippedRowsMap_;
  HypreIntTypeUnorderedMapHost skippedRowsMapHost_;
  HypreIntTypeUnorderedMap oversetRowsMap_;
  HypreIntTypeUnorderedMapHost oversetRowsMapHost_;
  HypreIntType num_mat_pts_to_assemble_total_owned_;
  HypreIntType num_mat_pts_to_assemble_total_shared_;
  HypreIntType num_rhs_pts_to_assemble_total_owned_;
  HypreIntType num_rhs_pts_to_assemble_total_shared_;
  HypreIntType num_mat_overset_pts_owned_;
  HypreIntType num_rhs_overset_pts_owned_;

  void fill_entity_to_row_mapping();
  void fill_device_data_structures();
  void fill_hids_columns(const unsigned numNodes, 
			 stk::mesh::Entity const * nodes,
			 std::vector<HypreIntType> & hids,
			 std::vector<HypreIntType> & columns);
  void fill_owned_shared_data_structures_1DoF(const unsigned numNodes, std::vector<HypreIntType>& hids);
  void fill_owned_shared_data_structures(const unsigned numNodes, std::vector<HypreIntType>& hids, 
					 std::vector<HypreIntType>& columns);

  // Quiet "partially overridden" compiler warnings.
  using LinearSystem::buildDirichletNodeGraph;
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
  virtual void buildNodeGraph(const stk::mesh::PartVector & parts);// for nodal assembly (e.g., lumped mass and source)
  virtual void buildFaceToNodeGraph(const stk::mesh::PartVector & parts);// face->node assembly
  virtual void buildEdgeToNodeGraph(const stk::mesh::PartVector & parts);// edge->node assembly
  virtual void buildElemToNodeGraph(const stk::mesh::PartVector & parts);// elem->node assembly
  virtual void buildReducedElemToNodeGraph(const stk::mesh::PartVector&);// elem (nearest nodes only)->node assembly
  virtual void buildFaceElemToNodeGraph(const stk::mesh::PartVector & parts);// elem:face->node assembly
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
  virtual void buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes);

  sierra::nalu::CoeffApplier* get_coeff_applier();

  /***************************************************************************************************/
  /*                     Beginning of HypreLinSysCoeffApplier definition                             */
  /***************************************************************************************************/

  class HypreLinSysCoeffApplier : public CoeffApplier
  {
  public:

    HypreLinSysCoeffApplier(bool useNativeCudaSort, bool ensureReproducible, unsigned numDof,
			    unsigned numDim, HypreIntType globalNumRows, int rank, 
			    HypreIntType iLower, HypreIntType iUpper,
			    HypreIntType jLower, HypreIntType jUpper,
			    MemoryMap mat_map_shared, HypreIntTypeView mat_elem_keys_owned,
			    UnsignedView mat_elem_start_owned, UnsignedView mat_elem_start_shared,
			    UnsignedView mat_row_start_owned, UnsignedView mat_row_start_shared,
			    MemoryMap rhs_map_shared, 
			    UnsignedView rhs_row_start_owned, UnsignedView rhs_row_start_shared,
			    HypreIntTypeView row_indices_owned, HypreIntTypeView row_indices_shared, 
			    HypreIntTypeView row_counts_owned, HypreIntTypeView row_counts_shared,
			    HypreIntType num_mat_pts_to_assemble_total_owned,
			    HypreIntType num_mat_pts_to_assemble_total_shared,
			    HypreIntType num_rhs_pts_to_assemble_total_owned,
			    HypreIntType num_rhs_pts_to_assemble_total_shared,
			    HypreIntTypeView periodic_bc_rows_owned,
			    HypreIntTypeView entityToLID, HypreIntTypeViewHost entityToLIDHost,
			    HypreIntTypeUnorderedMap skippedRowsMap, HypreIntTypeUnorderedMapHost skippedRowsMapHost,
			    HypreIntTypeUnorderedMap oversetRowsMap, HypreIntTypeUnorderedMapHost oversetRowsMapHost,
			    HypreIntType num_mat_overset_pts_owned, HypreIntType num_rhs_overset_pts_owned);

    KOKKOS_FUNCTION
    virtual ~HypreLinSysCoeffApplier() {
#ifdef HYPRE_LINEAR_SYSTEM_TIMER
      if (_nAssembleMat>0) {
	printf("\tMean HYPRE_IJMatrixSetValues Time (%d samples)=%1.5f   Total=%1.5f\n",
	       _nAssembleMat, _assembleMatTime/_nAssembleMat,_assembleMatTime);
	printf("\tMean HYPRE_IJVectorSetValues Time (%d samples)=%1.5f   Total=%1.5f\n",
	       _nAssembleRhs, _assembleRhsTime/_nAssembleRhs,_assembleRhsTime);
      }
#endif
    }

    KOKKOS_FUNCTION
    virtual void resetRows(unsigned,
                           const stk::mesh::Entity*,
                           const unsigned,
                           const unsigned,
                           const double,
                           const double) { 
      checkSkippedRows_() = 0;
    }
  
    KOKKOS_FUNCTION
    virtual void binarySearch(unsigned l, unsigned r, HypreIntType x, unsigned& result);

    KOKKOS_FUNCTION
    virtual void binarySearch(HypreIntTypeView view, unsigned l, unsigned r, HypreIntType x, unsigned& result);

    KOKKOS_FUNCTION
    virtual void sum_into(unsigned numEntities,
			  const stk::mesh::NgpMesh::ConnectedNodes& entities,
			  const SharedMemView<int*,DeviceShmem> & localIds,
			  const SharedMemView<const double*,DeviceShmem> & rhs,
			  const SharedMemView<const double**,DeviceShmem> & lhs,
			  const HypreIntType& iLower, const HypreIntType& iUpper,
			  unsigned numDof);

    KOKKOS_FUNCTION
    virtual void sum_into_1DoF(unsigned numEntities,
			       const stk::mesh::NgpMesh::ConnectedNodes& entities,
			       const SharedMemView<int*,DeviceShmem> & localIds,
			       const SharedMemView<const double*,DeviceShmem> & rhs,
			       const SharedMemView<const double**,DeviceShmem> & lhs,
			       const HypreIntType& iLower, const HypreIntType& iUpper);


    KOKKOS_FUNCTION
    virtual void operator()(unsigned numEntities,
                            const stk::mesh::NgpMesh::ConnectedNodes& entities,
                            const SharedMemView<int*,DeviceShmem> & localIds,
                            const SharedMemView<int*,DeviceShmem> &,
                            const SharedMemView<const double*,DeviceShmem> & rhs,
                            const SharedMemView<const double**,DeviceShmem> & lhs,
                            const char * trace_tag);

    virtual void sortMatrixElementBins(const HypreIntType nrows, const HypreIntType N,
				       const HypreIntType global_num_cols, 
				       const UnsignedView & mat_row_start,
				       const HypreIntTypeView & row_indices,
				       HypreIntTypeView & iwork,
				       HypreIntTypeView & col_indices_in_out,
				       DoubleView & values_in_out);


    virtual void fillCSRMatrix(const HypreIntType nnz, const HypreIntType N,
			       const UnsignedView & mat_elem_start,
			       const HypreIntTypeView & col_indices_in,
			       const DoubleView & values_in,
			       HypreIntTypeView & col_indices_out,
			       DoubleView & values_out);

    virtual void sortRhsElementBins(const HypreIntType nrows, const HypreIntType N, const unsigned index,
				    const HypreIntTypeView & row_indices, const UnsignedView & rhs_row_start,
				    HypreIntTypeView & iwork, DoubleView2D & values_in_out);
    
    virtual void fillRhsVector(const HypreIntType nrows, const HypreIntType N, const int index,
			       const UnsignedView & rhs_row_start,
			       const DoubleView2D & values_in,
			       DoubleView2D & values_out);

    virtual int nextPowerOfTwo(int v, int max) {
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v++;
      if (v>max) v=max;
      return v;
    }

    virtual void free_device_pointer();

    virtual sierra::nalu::CoeffApplier* device_pointer();

    virtual void resetInternalData();

    virtual void applyDirichletBCs(Realm & realm, 
				   stk::mesh::FieldBase * solutionField,
				   stk::mesh::FieldBase * bcValuesField,
				   const stk::mesh::PartVector& parts);
    
    virtual void finishAssembly(HYPRE_IJMatrix hypreMat, std::vector<HYPRE_IJVector> hypreRhs);
    
    virtual void sum_into_nonNGP(const std::vector<stk::mesh::Entity>& entities,
				 const std::vector<double>& rhs,
				 const std::vector<double>& lhs);

    //! whether or not to enforce reproducibility
    bool useNativeCudaSort_=false;
    //! whether or not to enforce reproducibility
    bool ensureReproducible_=false;
    //! number of degrees of freedom
    unsigned numDof_=0;
    //! number of rhs vectors
    unsigned nDim_=0;
    //! Maximum Row ID in the Hypre linear system
    HypreIntType globalNumRows_;
    //! Maximum Row ID in the Hypre linear system
    int rank_;
    //! The lowest row owned by this MPI rank
    HypreIntType iLower_=0;
    //! The highest row owned by this MPI rank
    HypreIntType iUpper_=0;
    //! The lowest column owned by this MPI rank; currently jLower_ == iLower_
    HypreIntType jLower_=0;
    //! The highest column owned by this MPI rank; currently jUpper_ == iUpper_
    HypreIntType jUpper_=0;

    //! map from dense index key to starting memory location ... shared
    MemoryMap mat_map_shared_;
    //! the matrix element keys ... owned
    HypreIntTypeView mat_elem_keys_owned_;
    //! the starting position(s) of the matrix element in the lists ... owned
    UnsignedView mat_elem_start_owned_;
    //! the starting position(s) of the matrix element in the lists ... shared
    UnsignedView mat_elem_start_shared_;
    //! the starting position(s) of a new row in the matrix lists ... owned
    UnsignedView mat_row_start_owned_;
    //! the starting position(s) of a new row in the matrix lists ... shared
    UnsignedView mat_row_start_shared_;
    //! map from rhs row index to starting memory location ... shared
    MemoryMap rhs_map_shared_;
    //! the starting position(s) of the rhs lists ... owned
    UnsignedView rhs_row_start_owned_;
    //! the starting position(s) of the rhs lists ... shared
    UnsignedView rhs_row_start_shared_;
    //! the row indices ... owned
    HypreIntTypeView row_indices_owned_;
    //! the row indices ... shared
    HypreIntTypeView row_indices_shared_;
    //! the row counts ... owned
    HypreIntTypeView row_counts_owned_;
    //! the row counts ... shared
    HypreIntTypeView row_counts_shared_;
    //! total number of points in the matrix owned lists
    HypreIntType num_mat_pts_to_assemble_total_owned_;
    //! total number of points in the matrix shared lists
    HypreIntType num_mat_pts_to_assemble_total_shared_;
    //! total number of points in the rhs owned lists
    HypreIntType num_rhs_pts_to_assemble_total_owned_;
    //! total number of points in the rhs shared lists
    HypreIntType num_rhs_pts_to_assemble_total_shared_;
      
    //! rows for the periodic boundary conditions ... owned. There is no shared version of this
    HypreIntTypeView periodic_bc_rows_owned_;

    //! A way to map the entity local offset to the hypre id
    HypreIntTypeView entityToLID_;
    HypreIntTypeViewHost entityToLIDHost_;

    //! unordered map for skipped rows
    HypreIntTypeUnorderedMap skippedRowsMap_;
    HypreIntTypeUnorderedMapHost skippedRowsMapHost_;

    //! unordered map for overset rows
    HypreIntTypeUnorderedMap oversetRowsMap_;
    HypreIntTypeUnorderedMapHost oversetRowsMapHost_;

    //! number of points in the overset data structures
    HypreIntType num_mat_overset_pts_owned_;
    HypreIntType num_rhs_overset_pts_owned_;

    //! this is the pointer to the device function ... that assembles the lists
    HypreLinSysCoeffApplier* devicePointer_;

    /* flag to reinitialize or not */
    bool reinitialize_=true;

    //! data structures to atomically update for augmenting the list */
    UnsignedView mat_counter_owned_;
    UnsignedView rhs_counter_owned_;
    UnsignedView mat_counter_shared_;
    UnsignedView rhs_counter_shared_;
    
    //! list for the column indices ... later to be assembled to the CSR matrix in Hypre
    HypreIntTypeView cols_owned_;
    HypreIntTypeView cols_shared_;

    //! list for the values ... later to be assembled to the CSR matrix in Hypre
    DoubleView vals_owned_;
    DoubleView vals_shared_;

    //! list for the rhs values ... later to be assembled to the rhs vector in Hypre
    DoubleView2D rhs_vals_owned_;
    DoubleView2D rhs_vals_shared_;

    //! Total number of rows owned by this particular MPI rank
    HypreIntType num_rows_;
    HypreIntType num_rows_owned_;
    HypreIntType num_rows_shared_;
    DoubleView d_values_owned_;
    DoubleViewHost h_values_owned_;
    HypreIntTypeView d_col_indices_owned_;
    HypreIntTypeViewHost h_col_indices_owned_;
    HypreIntTypeViewHost h_row_indices_owned_;
    HypreIntTypeViewHost h_row_counts_owned_;
    DoubleView2D d_rhs_owned_;
    DoubleView2DHost h_rhs_owned_;

    //! Total number of rows shared by this particular MPI rank
    HypreIntType num_nonzeros_;
    HypreIntType num_nonzeros_owned_;
    HypreIntType num_nonzeros_shared_;
    DoubleView d_values_shared_;
    DoubleViewHost h_values_shared_;
    HypreIntTypeView d_col_indices_shared_;
    HypreIntTypeViewHost h_col_indices_shared_;
    HypreIntTypeViewHost h_row_indices_shared_;
    HypreIntTypeViewHost h_row_counts_shared_;
    DoubleView2D d_rhs_shared_;
    DoubleView2DHost h_rhs_shared_;

    //! Flag indicating that sumInto should check to see if rows must be skipped
    HypreIntTypeViewScalar checkSkippedRows_;

    /* Work space */
    HypreIntTypeView iwork_;
    DoubleView dwork_;

    /* Work space for overset. These are used to accumulate data from legacy, non-NGP sumInto calls */
    HypreIntTypeView d_overset_row_indices_;
    HypreIntTypeViewHost h_overset_row_indices_;

    HypreIntTypeView d_overset_keys_;
    HypreIntTypeViewHost h_overset_keys_;

    DoubleView d_overset_vals_;
    DoubleViewHost h_overset_vals_;

    DoubleView d_overset_rhs_vals_;
    DoubleViewHost h_overset_rhs_vals_;

    /* counters for adding to the array */
    int overset_mat_counter_=0;
    int overset_rhs_counter_=0;

    float _assembleMatTime=0.f;
    float _assembleRhsTime=0.f;
    int _nAssembleMat=0;
    int _nAssembleRhs=0;
  };

  /***************************************************************************************************/
  /*                        End of of HypreLinSysCoeffApplier definition                             */
  /***************************************************************************************************/

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
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const SharedMemView<const double*, DeviceShmem> & rhs,
    const SharedMemView<const double**, DeviceShmem> & lhs,
    const SharedMemView<int*, DeviceShmem> & localIds,
    const SharedMemView<int*, DeviceShmem> & sortPermutation,
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

  /** Dummy method to satisfy inheritance
   */
  void checkError(
    const int,
    const char*) {}

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
  bool matrixAssembled_{false};

  //! Flag indicating whether the linear system has been initialized
  bool systemInitialized_{false};

private:
  //! HYPRE right hand side data structure
  mutable HYPRE_IJVector rhs_;

  //! HYPRE solution vector
  mutable HYPRE_IJVector sln_;

};

}  // nalu
}  // sierra


#endif /* HYPRELINEARSYSTEM_H */
