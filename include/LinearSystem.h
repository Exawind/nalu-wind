// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef LinearSystem_h
#define LinearSystem_h

#include <LinearSolverTypes.h>
#include <KokkosInterface.h>

#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpMesh.hpp>

#include <vector>
#include <string>

namespace stk {
namespace mesh {
struct Entity;
}
} // namespace stk

namespace stk {
namespace mesh {
class FieldBase;
class Part;
typedef std::vector<Part*> PartVector;
} // namespace mesh
} // namespace stk

namespace sierra {
namespace nalu {

class EquationSystem;
class Realm;
class LinearSolver;
class LinearSolverConfig;

class CoeffApplier
{
public:
  KOKKOS_DEFAULTED_FUNCTION
  CoeffApplier() = default;

  KOKKOS_DEFAULTED_FUNCTION
  virtual ~CoeffApplier() = default;

  KOKKOS_FUNCTION
  virtual void resetRows(
    unsigned numNodes,
    const stk::mesh::Entity* nodeList,
    const unsigned beginPos,
    const unsigned endPos,
    const double diag_value = 0.0,
    const double rhs_residual = 0.0) = 0;

  KOKKOS_FUNCTION
  virtual void operator()(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const SharedMemView<int*, DeviceShmem>& localIds,
    const SharedMemView<int*, DeviceShmem>& sortPermutation,
    const SharedMemView<const double*, DeviceShmem>& rhs,
    const SharedMemView<const double**, DeviceShmem>& lhs,
    const char* trace_tag) = 0;

  virtual void free_device_pointer() = 0;
  virtual CoeffApplier* device_pointer() = 0;
};

struct CoeffApplierDestructor
{
  using ptr_t = CoeffApplier;

  void free(ptr_t* coeffApplier)
  {
    if (coeffApplier != nullptr) {
      kokkos_free_on_device(coeffApplier);
    }
  }
};

class LinearSystem
{
public:
  LinearSystem(
    Realm& realm,
    const unsigned numDof,
    EquationSystem* eqSys,
    LinearSolver* linearSolver);

  virtual ~LinearSystem()
  {
    if (hostCoeffApplier) {
      hostCoeffApplier->free_device_pointer();
      deviceCoeffApplier = nullptr;
    }
  }

  static LinearSystem* create(
    Realm& realm,
    const unsigned numDof,
    EquationSystem* eqSys,
    LinearSolver* linearSolver);

  const LinearSolverConfig& config() const;

  // Graph/Matrix Construction
  virtual void buildNodeGraph(
    const stk::mesh::PartVector&
      parts) = 0; // for nodal assembly (e.g., lumped mass and source)
  virtual void buildFaceToNodeGraph(
    const stk::mesh::PartVector& parts) = 0; // face->node assembly
  virtual void buildEdgeToNodeGraph(
    const stk::mesh::PartVector& parts) = 0; // edge->node assembly
  virtual void buildElemToNodeGraph(
    const stk::mesh::PartVector& parts) = 0; // elem->node assembly
  virtual void buildReducedElemToNodeGraph(
    const stk::mesh::PartVector&
      parts) = 0; // elem (nearest nodes only)->node assembly
  virtual void buildFaceElemToNodeGraph(
    const stk::mesh::PartVector& parts) = 0; // elem:face->node assembly
  virtual void buildNonConformalNodeGraph(
    const stk::mesh::PartVector& parts) = 0; // nonConformal->elem_node assembly
  virtual void buildOversetNodeGraph(
    const stk::mesh::PartVector& parts) = 0; // overset->elem_node assembly
  virtual void finalizeLinearSystem() = 0;

  /** Process nodes that belong to Dirichlet-type BC
   *
   */
  virtual void buildDirichletNodeGraph(const stk::mesh::PartVector&) {}

  /** Process nodes as belonging to a Dirichlet-type row
   *
   *  See the documentation/implementation of
   *  sierra::nalu::FixPressureAtNodeAlgorithm for an example of this use case.
   */
  virtual void buildDirichletNodeGraph(const std::vector<stk::mesh::Entity>&) {}
  virtual void buildDirichletNodeGraph(const stk::mesh::NgpMesh::ConnectedNodes)
  {
  }

  virtual void free_coeff_applier(CoeffApplier*) {}

  // Matrix Assembly
  virtual void zeroSystem() = 0;

  virtual CoeffApplier* get_coeff_applier() { return nullptr; }

  virtual bool owns_coeff_applier() { return true; }

  virtual void sumInto(
    unsigned numEntities,
    const stk::mesh::NgpMesh::ConnectedNodes& entities,
    const SharedMemView<const double*, DeviceShmem>& rhs,
    const SharedMemView<const double**, DeviceShmem>& lhs,
    const SharedMemView<int*, DeviceShmem>& localIds,
    const SharedMemView<int*, DeviceShmem>& sortPermutation,
    const char* trace_tag) = 0;

  virtual void sumInto(
    const std::vector<stk::mesh::Entity>& sym_meshobj,
    std::vector<int>& scratchIds,
    std::vector<double>& scratchVals,
    const std::vector<double>& rhs,
    const std::vector<double>& lhs,
    const char* trace_tag = 0) = 0;

  virtual void applyDirichletBCs(
    stk::mesh::FieldBase* solutionField,
    stk::mesh::FieldBase* bcValuesField,
    const stk::mesh::PartVector& parts,
    const unsigned beginPos,
    const unsigned endPos) = 0;

  /** Reset LHS and RHS for the given set of nodes to 0
   *
   *  @param nodeList A list of STK node entities whose rows are zeroed out
   *  @param beginPos Starting index (usually 0)
   *  @param endPos Terminating index (1 for scalar quantities; nDim for
   * vectors)
   */
  virtual void resetRows(
    const std::vector<stk::mesh::Entity>& nodeList,
    const unsigned beginPos,
    const unsigned endPos,
    const double diag_value = 0.0,
    const double rhs_residual = 0.0) = 0;

  virtual void resetRows(
    unsigned numNodes,
    const stk::mesh::Entity* nodeList,
    const unsigned beginPos,
    const unsigned endPos,
    const double diag_value = 0.0,
    const double rhs_residual = 0.0) = 0;

  // Solve
  virtual int solve(stk::mesh::FieldBase* linearSolutionField) = 0;
  virtual void loadComplete() = 0;

  virtual void writeToFile(const char* filename, bool useOwned = true) = 0;
  virtual void
  writeSolutionToFile(const char* filename, bool useOwned = true) = 0;
  virtual unsigned numDof() const { return numDof_; }
  const int& linearSolveIterations() const { return linearSolveIterations_; }
  const double& linearResidual() const { return linearResidual_; }
  const double& nonLinearResidual() const { return nonLinearResidual_; }
  const double& scaledNonLinearResidual() const
  {
    return scaledNonLinearResidual_;
  }
  void setNonLinearResidual(const double nlr) { nonLinearResidual_ = nlr; }
  std::string name() const { return eqSysName_; }
  bool recomputePreconditioner() const { return recomputePreconditioner_; }
  bool reusePreconditioner() const { return reusePreconditioner_; }
  double get_timer_precond();
  void zero_timer_precond();
  bool useSegregatedSolver() const;

  EquationSystem* equationSystem() { return eqSys_; }

protected:
  virtual void beginLinearSystemConstruction() = 0;
  virtual void checkError(const int err_code, const char* msg) = 0;

  void sync_field(const stk::mesh::FieldBase* field);
  bool debug();

  Realm& realm_;
  EquationSystem* eqSys_;
  bool inConstruction_;

  const unsigned numDof_;
  const std::string eqSysName_;
  LinearSolver* linearSolver_{nullptr};
  int linearSolveIterations_;
  double nonLinearResidual_;
  double linearResidual_;
  double firstNonLinearResidual_;
  double scaledNonLinearResidual_;
  bool recomputePreconditioner_;
  bool reusePreconditioner_;

  std::unique_ptr<CoeffApplier> hostCoeffApplier;
  CoeffApplier* deviceCoeffApplier = nullptr;

public:
  bool provideOutput_;
};

} // namespace nalu
} // namespace sierra

#endif
