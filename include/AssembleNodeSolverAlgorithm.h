// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleNodeSolverAlgorithm_h
#define AssembleNodeSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodeSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleNodeSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);

  virtual ~AssembleNodeSolverAlgorithm() = default;
  virtual void initialize_connectivity();
  virtual void execute();

  const int sizeOfSystem_;
};

} // namespace nalu
} // namespace Sierra

#endif
