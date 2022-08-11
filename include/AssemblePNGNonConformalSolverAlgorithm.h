// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef AssemblePNGNonConformalSolverAlgorithm_h
#define AssemblePNGNonConformalSolverAlgorithm_h

#include <SolverAlgorithm.h>
#include <FieldTypeDef.h>

// stk
#include <stk_mesh/base/Part.hpp>

namespace sierra {
namespace nalu {

class Realm;

class AssemblePNGNonConformalSolverAlgorithm : public SolverAlgorithm
{
public:
  AssemblePNGNonConformalSolverAlgorithm(
    Realm& realm,
    stk::mesh::Part* part,
    EquationSystem* eqSystem,
    std::string independentDofName,
    std::string dofName,
    const bool includePenalty);

  ~AssemblePNGNonConformalSolverAlgorithm();

  virtual void initialize_connectivity();
  virtual void execute();

  ScalarFieldType* scalarQ_;
  VectorFieldType* Gjq_;
  VectorFieldType* coordinates_;
  GenericFieldType* exposedAreaVec_;

  // options that prevail over all algorithms created
  const double useCurrentNormal_;
  const double includePenalty_;
  std::vector<const stk::mesh::FieldBase*> ghostFieldVec_;
};

} // namespace nalu
} // namespace sierra

#endif
