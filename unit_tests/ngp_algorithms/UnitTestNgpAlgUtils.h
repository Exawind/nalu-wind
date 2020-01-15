// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


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
