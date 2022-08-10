// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef UNITTESTFIELDUTILS_H
#define UNITTESTFIELDUTILS_H

#include <gtest/gtest.h>
#include "UnitTestUtils.h"
#include "UnitTestKokkosUtils.h"

namespace unit_test_utils {

double field_norm(
  const ScalarFieldType& field,
  const stk::mesh::BulkData& bulk,
  stk::mesh::Selector selector);

}

#endif /* UNITTESTFIELDUTILS_H */
