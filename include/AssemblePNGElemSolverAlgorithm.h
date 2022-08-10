// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssemblePNGElemSolverAlgorithm_h
#define AssemblePNGElemSolverAlgorithm_h

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

class AssemblePNGElemSolverAlgorithm : public SolverAlgorithm
{
public:
  AssemblePNGElemSolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    std::string independentDofName,
    std::string dofName);
  virtual ~AssemblePNGElemSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  ScalarFieldType* scalarQ_;
  VectorFieldType* dqdx_;
  VectorFieldType* coordinates_;
};

} // namespace nalu
} // namespace sierra

#endif
