// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleMomentumElemOpenSolverAlgorithm_h
#define AssembleMomentumElemOpenSolverAlgorithm_h

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
template <typename T> class PecletFunction;

class AssembleMomentumElemOpenSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleMomentumElemOpenSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);
  virtual ~AssembleMomentumElemOpenSolverAlgorithm();
  virtual void initialize_connectivity();
  virtual void execute();

  const double includeDivU_;
  const bool meshMotion_;

  VectorFieldType *velocityRTM_;
  VectorFieldType *velocity_;
  GenericFieldType *dudx_;
  VectorFieldType *coordinates_;
  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  GenericFieldType *exposedAreaVec_;
  GenericFieldType *openMassFlowRate_;
  VectorFieldType *velocityBc_;

  // peclet function specifics
  PecletFunction<double>* pecletFunction_;
};

} // namespace nalu
} // namespace Sierra

#endif
