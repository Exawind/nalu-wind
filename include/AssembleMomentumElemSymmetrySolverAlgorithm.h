// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleMomentumElemSymmetrySolverAlgorithm_h
#define AssembleMomentumElemSymmetrySolverAlgorithm_h

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

class AssembleMomentumElemSymmetrySolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleMomentumElemSymmetrySolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);
  virtual ~AssembleMomentumElemSymmetrySolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const double includeDivU_;

  VectorFieldType *velocity_;
  VectorFieldType *coordinates_;
  ScalarFieldType *viscosity_;
  GenericFieldType *exposedAreaVec_;
};

} // namespace nalu
} // namespace Sierra

#endif
