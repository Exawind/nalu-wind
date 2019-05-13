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

  /** Reset rows for the holes in the linear system
   *
   *  This method will call reset rows and then populate them with a 1 on the
   *  diagonal and 0 on the RHS entry for the row.
   */
  void reset_hole_rows();

  stk::mesh::FieldBase* fieldQ_;
  ScalarFieldType* dualNodalVolume_{nullptr};
};

}  // nalu
}  // sierra


#endif /* OVERSETCONSTRAINTBASE_H */
