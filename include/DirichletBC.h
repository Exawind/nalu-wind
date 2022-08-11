// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef DirichletBC_h
#define DirichletBC_h

#include <SolverAlgorithm.h>

namespace stk {
namespace mesh {
class FieldBase;
}
} // namespace stk

namespace sierra {
namespace nalu {

class EquationSystem;
class Realm;

class DirichletBC : public SolverAlgorithm
{
public:
  DirichletBC(
    Realm& realm,
    EquationSystem* eqSystem,
    stk::mesh::Part* part,
    stk::mesh::FieldBase* field,
    stk::mesh::FieldBase* bcValues,
    const unsigned beginPos,
    const unsigned endPos);

  virtual ~DirichletBC() {}

  virtual void execute();
  virtual void initialize_connectivity();

private:
  stk::mesh::FieldBase* field_;
  stk::mesh::FieldBase* bcValues_;
  const unsigned beginPos_;
  const unsigned endPos_;

private:
  // make this non-copyable
  DirichletBC(const DirichletBC& other);
  DirichletBC& operator=(const DirichletBC& other);
};

} // namespace nalu
} // namespace sierra

#endif
