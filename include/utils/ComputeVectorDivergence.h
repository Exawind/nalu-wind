/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

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
