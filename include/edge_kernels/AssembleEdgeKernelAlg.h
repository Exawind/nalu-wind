// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ASSEMBLEEDGEKERNEL_H
#define ASSEMBLEEDGEKERNEL_H

#include "AssembleEdgeSolverAlgorithm.h"
#include "nalu_make_unique.h"

#include <vector>
#include <memory>

namespace sierra {
namespace nalu {

class Realm;
class EdgeKernel;

class AssembleEdgeKernelAlg : public AssembleEdgeSolverAlgorithm
{
public:
  using EdgeKernelPtrType = std::unique_ptr<EdgeKernel>;
  using EdgeKernelVecType = std::vector<EdgeKernelPtrType>;

  AssembleEdgeKernelAlg(Realm&, stk::mesh::Part*, EquationSystem*);

  virtual ~AssembleEdgeKernelAlg();

  virtual void execute() override;

  template <typename T, class... Args>
  void add_kernel(Args&&... args)
  {
    edgeKernels_.push_back(make_unique<T>(std::forward<Args>(args)...));
  }

protected:
  EdgeKernelVecType edgeKernels_;
};

} // namespace nalu
} // namespace sierra

#endif /* ASSEMBLEEDGEKERNEL_H */
