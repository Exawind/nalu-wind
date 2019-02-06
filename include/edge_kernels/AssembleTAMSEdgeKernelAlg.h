/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef ASSEMBLETAMSEDGEKERNEL_H
#define ASSEMBLETAMSEDGEKERNEL_H

#include "AssembleEdgeSolverAlgorithm.h"
#include "MomentumSSTTAMSDiffEdgeKernel.h"
#include "AssembleEdgeKernelAlg.h"
#include "nalu_make_unique.h"

#include <vector>
#include <memory>

namespace sierra {
namespace nalu {

class Realm;

class AssembleTAMSEdgeKernelAlg : public AssembleEdgeKernelAlg
{
  public:
    AssembleTAMSEdgeKernelAlg(Realm&, stk::mesh::Part*, EquationSystem*);
};

} // namespace nalu
} // namespace sierra

#endif /* ASSEMBLETAMSEDGEKERNEL_H */
