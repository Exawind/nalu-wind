// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleMomentumEdgeOpenSolverAlgorithm_h
#define AssembleMomentumEdgeOpenSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include <FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleMomentumEdgeOpenSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleMomentumEdgeOpenSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);
  virtual ~AssembleMomentumEdgeOpenSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const double includeDivU_;

  // extract fields
  VectorFieldType *velocity_;
  GenericFieldType *dudx_;
  VectorFieldType *coordinates_;
  ScalarFieldType *viscosity_;
  GenericFieldType *exposedAreaVec_;
  GenericFieldType *openMassFlowRate_;
  VectorFieldType *velocityBc_;
};

} // namespace nalu
} // namespace Sierra

#endif
