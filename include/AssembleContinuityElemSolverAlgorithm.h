// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleContinuityElemSolverAlgorithm_h
#define AssembleContinuityElemSolverAlgorithm_h

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

class AssembleContinuityElemSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleContinuityElemSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);
  virtual ~AssembleContinuityElemSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const bool meshMotion_;

  // extract fields; nodal
  VectorFieldType *velocityRTM_;
  VectorFieldType *Gpdx_;
  VectorFieldType *coordinates_;
  ScalarFieldType *pressure_;
  ScalarFieldType *density_;

  const bool shiftMdot_;
  const bool shiftPoisson_;
  const bool reducedSensitivities_;

};

} // namespace nalu
} // namespace Sierra

#endif
