// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleContinuityInflowSolverAlgorithm_h
#define AssembleContinuityInflowSolverAlgorithm_h

#include <SolverAlgorithm.h>
#include <FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class AssembleContinuityInflowSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleContinuityInflowSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    bool useShifted = false);
  virtual ~AssembleContinuityInflowSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const bool useShifted_;

  GenericFieldType *exposedAreaVec_;
  VectorFieldType *velocityBC_;
  ScalarFieldType *densityBC_;
};

} // namespace nalu
} // namespace Sierra

#endif
