// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleScalarDiffNonConformalSolverAlgorithm_h
#define AssembleScalarDiffNonConformalSolverAlgorithm_h

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

class AssembleScalarDiffNonConformalSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleScalarDiffNonConformalSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    ScalarFieldType *scalarQ,
    ScalarFieldType *diffFluxCoeff);
  virtual ~AssembleScalarDiffNonConformalSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  ScalarFieldType *scalarQ_;
  ScalarFieldType *diffFluxCoeff_;
  VectorFieldType *coordinates_;
  GenericFieldType *exposedAreaVec_;

  // options that prevail over all algorithms created
  const bool useCurrentNormal_;

  std::vector< const stk::mesh::FieldBase *> ghostFieldVec_;
};

} // namespace nalu
} // namespace Sierra

#endif
