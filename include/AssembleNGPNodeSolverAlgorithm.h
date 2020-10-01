// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ASSEMBLENGPNODESOLVERALGORITHM_H
#define ASSEMBLENGPNODESOLVERALGORITHM_H

#include "SolverAlgorithm.h"

#include <vector>
#include <memory>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra {
namespace nalu {

class Realm;
class NodeKernel;

class AssembleNGPNodeSolverAlgorithm : public SolverAlgorithm
{
public:
  using NodeKernelPtrType = std::unique_ptr<NodeKernel>;
  using NodeKernelVecType = std::vector<NodeKernelPtrType>;

  AssembleNGPNodeSolverAlgorithm(
    Realm&,
    stk::mesh::Part*,
    EquationSystem*);

  AssembleNGPNodeSolverAlgorithm() = delete;
  AssembleNGPNodeSolverAlgorithm(const AssembleNGPNodeSolverAlgorithm&) = delete;

  virtual ~AssembleNGPNodeSolverAlgorithm();

  virtual void initialize_connectivity() override;

  virtual void execute() override;

  template<typename T, class... Args>
  void add_kernel(Args&&... args)
  {
    nodeKernels_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

private:
  //! List of NodeKernels registered with this algorithm
  NodeKernelVecType nodeKernels_;

  //! Number of DOFs per nodal entity
  const int rhsSize_;
};

}  // nalu
}  // sierra



#endif /* ASSEMBLENGPNODESOLVERALGORITHM_H */
