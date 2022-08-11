// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssemblePNGBoundarySolverAlgorithm_h
#define AssemblePNGBoundarySolverAlgorithm_h

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

class AssemblePNGBoundarySolverAlgorithm : public SolverAlgorithm
{
public:
  AssemblePNGBoundarySolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    std::string independentDofName);
  virtual ~AssemblePNGBoundarySolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  ScalarFieldType* scalarQ_;
  GenericFieldType* exposedAreaVec_;
};

} // namespace nalu
} // namespace sierra

#endif
