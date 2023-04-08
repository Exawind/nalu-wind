// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleMeshDisplacementElemSolverAlgorithm_h
#define AssembleMeshDisplacementElemSolverAlgorithm_h

#include <SolverAlgorithm.h>
#include <FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
}
} // namespace stk

namespace sierra {
namespace nalu {

class Realm;

class AssembleMeshDisplacementElemSolverAlgorithm : public SolverAlgorithm
{
public:
  AssembleMeshDisplacementElemSolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    const bool deformWrtModelCoords);
  virtual ~AssembleMeshDisplacementElemSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const bool deformWrtModelCoords_;
  VectorFieldType* meshDisplacement_;
  VectorFieldType* coordinates_;
  VectorFieldType* modelCoordinates_;
  ScalarFieldType* mu_;
  ScalarFieldType* lambda_;
};

} // namespace nalu
} // namespace sierra

#endif
