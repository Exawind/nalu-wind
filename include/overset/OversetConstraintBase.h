// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef OVERSETCONSTRAINTBASE_H
#define OVERSETCONSTRAINTBASE_H

#include "SolverAlgorithm.h"
#include "FieldTypeDef.h"

namespace sierra {
namespace nalu {

class OversetConstraintBase : public SolverAlgorithm
{
public:
  OversetConstraintBase(
    Realm&, stk::mesh::Part*, EquationSystem*, stk::mesh::FieldBase*);

  virtual ~OversetConstraintBase() = default;

  virtual void initialize_connectivity();

  /** Reset rows for the holes in the linear system
   *
   *  This method will call reset rows and then populate them with a 1 on the
   *  diagonal and 0 on the RHS entry for the row.
   */
  virtual void prepare_constraints();

protected:
  OversetConstraintBase() = delete;
  OversetConstraintBase(const OversetConstraintBase&) = delete;

  stk::mesh::FieldBase* fieldQ_;
  ScalarFieldType* dualNodalVolume_{nullptr};
};

} // namespace nalu
} // namespace sierra

#endif /* OVERSETCONSTRAINTBASE_H */
