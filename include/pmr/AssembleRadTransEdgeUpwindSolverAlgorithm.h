// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//



#ifndef AssembleRadTransEdgeUpwindSolverAlgorithm_h
#define AssembleRadTransEdgeUpwindSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class RadiativeTransportEquationSystem;
class Realm;

class AssembleRadTransEdgeUpwindSolverAlgorithm : public SolverAlgorithm
{
public:
  
  AssembleRadTransEdgeUpwindSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    RadiativeTransportEquationSystem *radEqSystem);
  virtual ~AssembleRadTransEdgeUpwindSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const RadiativeTransportEquationSystem *radEqSystem_;

  ScalarFieldType *intensity_;
  VectorFieldType *edgeAreaVec_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
