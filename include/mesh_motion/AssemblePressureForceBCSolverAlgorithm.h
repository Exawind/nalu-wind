// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef AssemblePressureForceBCSolverAlgorithm_h
#define AssemblePressureForceBCSolverAlgorithm_h

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

class AssemblePressureForceBCSolverAlgorithm : public SolverAlgorithm
{
public:

  AssemblePressureForceBCSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    ScalarFieldType *bcScalarQ,
    bool use_shifted_integration);
  virtual ~AssemblePressureForceBCSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

private:

  ScalarFieldType *bcScalarQ_;
  VectorFieldType *coordinates_;
  GenericFieldType *exposedAreaVec_;

  bool use_shifted_integration_;

};

} // namespace nalu
} // namespace Sierra

#endif
