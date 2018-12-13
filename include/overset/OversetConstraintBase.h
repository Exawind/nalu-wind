/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
    Realm&,
    stk::mesh::Part*,
    EquationSystem*,
    stk::mesh::FieldBase*);

  virtual ~OversetConstraintBase() = default;

  virtual void initialize_connectivity();

  virtual void prepare_constraints();

protected:
  OversetConstraintBase() = delete;
  OversetConstraintBase(const OversetConstraintBase&) = delete;

  stk::mesh::FieldBase* fieldQ_;
  ScalarFieldType* dualNodalVolume_{nullptr};
};

}  // nalu
}  // sierra


#endif /* OVERSETCONSTRAINTBASE_H */
