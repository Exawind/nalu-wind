// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleHeatCondWallSolverAlgorithm_h
#define AssembleHeatCondWallSolverAlgorithm_h

#include <SolverAlgorithm.h>
#include <FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleHeatCondWallSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleHeatCondWallSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    ScalarFieldType *referenceTemp,
    ScalarFieldType *couplingParameter,
    ScalarFieldType *normalHeatFlux,
    bool useShifted = false);
  virtual ~AssembleHeatCondWallSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const bool useShifted_;

  GenericFieldType *exposedAreaVec_;
  ScalarFieldType *referenceTemp_;
  ScalarFieldType *couplingParameter_;
  ScalarFieldType *normalHeatFlux_;
  ScalarFieldType *temperature_;
};

} // namespace nalu
} // namespace Sierra

#endif
