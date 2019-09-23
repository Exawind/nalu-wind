/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef UNITTESTNGPALGUTILS_H
#define UNITTESTNGPALGUTILS_H

#include "UnitTestUtils.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/BulkData.hpp"

namespace unit_test_alg_utils
{
void linear_scalar_field(
  const stk::mesh::BulkData&, const VectorFieldType&, ScalarFieldType&,
  const double xCoeff=1.0, const double yCoeff=1.0, const double zCoeff=1.0);

void linear_scalar_field(
  const stk::mesh::BulkData&, const VectorFieldType&, VectorFieldType&,
  const double xCoeff=1.0, const double yCoeff=1.0, const double zCoeff=1.0);
}

#endif /* UNITTESTNGPALGUTILS_H */
