// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef ASSEMBLEAMSEDGEKERNEL_H
#define ASSEMBLEAMSEDGEKERNEL_H

#include "AssembleEdgeKernelAlg.h"

namespace sierra {
namespace nalu {

class Realm;

class AssembleAMSEdgeKernelAlg : public AssembleEdgeKernelAlg
{
public:
  AssembleAMSEdgeKernelAlg(Realm&, stk::mesh::Part*, EquationSystem*);
};

} // namespace nalu
} // namespace sierra

#endif /* ASSEMBLEAMSEDGEKERNEL_H */
