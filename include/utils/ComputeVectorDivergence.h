// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef COMPUTEVECTORDIVERGENCE_H
#define COMPUTEVECTORDIVERGENCE_H

#include <FieldTypeDef.h>

namespace stk {
namespace mesh {
  class BulkData;
  class Part;
  typedef std::vector<Part*> PartVector;
}
}

namespace sierra {
namespace nalu {

void compute_vector_divergence(
  stk::mesh::BulkData&,
  stk::mesh::PartVector&,
  stk::mesh::PartVector&,
  stk::mesh::FieldBase*,
  stk::mesh::FieldBase*,
  const bool hasMeshDeformation = false );

}
}

#endif /* COMPUTEVECTORDIVERGENCE_H */
