/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
