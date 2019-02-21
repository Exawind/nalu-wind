/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTEDIVERGENCE_H
#define COMPUTEDIVERGENCE_H

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

class ComputeDivergence
{

public:
  ComputeDivergence();
  ~ComputeDivergence();

  void operate(
    stk::mesh::BulkData &,
    stk::mesh::PartVector &,
    stk::mesh::FieldBase *,
    stk::mesh::FieldBase * );
};
    
}
}

#endif /* COMPUTEDIVERGENCE_H */
