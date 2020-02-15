// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef ASSEMBLETAMSEDGEKERNEL_H
#define ASSEMBLETAMSEDGEKERNEL_H

#include "AssembleEdgeKernelAlg.h"

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
