// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssembleMomentumNonConformalSolverAlgorithm_h
#define AssembleMomentumNonConformalSolverAlgorithm_h

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

class AssembleMomentumNonConformalSolverAlgorithm : public SolverAlgorithm
{
public:
  AssembleMomentumNonConformalSolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    VectorFieldType* velocity,
    ScalarFieldType* diffFluxCoeff);
  virtual ~AssembleMomentumNonConformalSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  VectorFieldType* velocity_;
  ScalarFieldType* diffFluxCoeff_;
  VectorFieldType* coordinates_;
  GenericFieldType* exposedAreaVec_;
  GenericFieldType* ncMassFlowRate_;

  // options that prevail over all algorithms created
  const double eta_;
  const double includeDivU_;
  const double useCurrentNormal_;

  std::vector<const stk::mesh::FieldBase*> ghostFieldVec_;
};

} // namespace nalu
} // namespace sierra

#endif
