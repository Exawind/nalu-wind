// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef AssembleScalarFluxBCSolverAlgorithm_h
#define AssembleScalarFluxBCSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class LinearSystem;
class Realm;

class AssembleScalarFluxBCSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleScalarFluxBCSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    ScalarFieldType *bcScalarQ,
    bool useShifted);
  virtual ~AssembleScalarFluxBCSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

private:

  const bool useShifted_;

  ScalarFieldType *bcScalarQ_;
  GenericFieldType *exposedAreaVec_;
};

}
}


#endif /* ASSEMBLESCALARELEMDIFFBCSOLVERALGORITHM_H_ */
