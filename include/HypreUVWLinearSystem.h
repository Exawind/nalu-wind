// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef HYPREUVWLINEARSYSTEM_H
#define HYPREUVWLINEARSYSTEM_H

#include "HypreLinearSystem.h"
#include "HypreUVWSolver.h"

namespace sierra {
namespace nalu {

class HypreUVWLinearSystem : public HypreLinearSystem
{
public:
  HypreUVWLinearSystem(
    Realm& realm,
    const unsigned numDof,
    EquationSystem* eqSys,
    LinearSolver* linearSolver);

  virtual ~HypreUVWLinearSystem();

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
  // overset->elem_node assembly
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

  virtual void loadComplete();

  virtual void zeroSystem();

  virtual int solve(stk::mesh::FieldBase*);

  void copy_hypre_to_stk(stk::mesh::FieldBase*, std::vector<double>&);

  /** Populate the LHS and RHS for the Dirichlet rows in linear system
   */
  virtual void applyDirichletBCs(
    stk::mesh::FieldBase* solutionField,
    stk::mesh::FieldBase* bcValuesField,
    const stk::mesh::PartVector& parts,
    const unsigned beginPos,
    const unsigned endPos);

  virtual unsigned numDof() const { return nDim_; }

  sierra::nalu::CoeffApplier* get_coeff_applier();

  /***************************************************************************************************/
  /*                     Beginning of HypreLinSysCoeffApplier definition */
  /***************************************************************************************************/

  class HypreUVWLinSysCoeffApplier : public HypreLinSysCoeffApplier
  {
  public:
    HypreUVWLinSysCoeffApplier(
      unsigned numDof, unsigned nDim, HypreIntType iLower, HypreIntType iUpper);

    KOKKOS_DEFAULTED_FUNCTION
    virtual ~HypreUVWLinSysCoeffApplier() = default;

    KOKKOS_FUNCTION
    virtual void reset_rows(
      unsigned numNodes,
      const stk::mesh::Entity* nodeList,
      const double diag_value,
      const double rhs_residual,
      const HypreIntType iLower,
      const HypreIntType iUpper,
      const unsigned nDim,
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
    virtual void sum_into(
      unsigned numEntities,
      const stk::mesh::NgpMesh::ConnectedNodes& entities,
      const SharedMemView<int*, DeviceShmem>& localIds,
      const SharedMemView<int*, DeviceShmem>& sortPermutation,
      const SharedMemView<const double*, DeviceShmem>& rhs,
      const SharedMemView<const double**, DeviceShmem>& lhs,
      const HypreIntType& iLower,
      const HypreIntType& iUpper,
      unsigned nDim,
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
  };

  virtual void hypreIJVectorSetAddToValues();

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
    const std::vector<stk::mesh::Entity>& /*sym_meshobj*/,
    std::vector<int>& /*scratchIds*/,
    std::vector<double>& /*scratchVals*/,
    const std::vector<double>& /*rhs*/,
    const std::vector<double>& /*lhs*/,
    const char* /*trace_tag*/)
  {
  }

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

protected:
  virtual void finalizeLinearSystem();

  virtual void loadCompleteSolver();

private:
  std::vector<std::string> vecNames_{"X", "Y", "Z"};

  std::array<double, 3> firstNLR_;

  mutable std::vector<HYPRE_IJVector> rhs_;
  mutable std::vector<HYPRE_IJVector> sln_;

  const unsigned nDim_{3};
};

} // namespace nalu
} // namespace sierra

#endif /* HYPREUVWLINEARSYSTEM_H */
