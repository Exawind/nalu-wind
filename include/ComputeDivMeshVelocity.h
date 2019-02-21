/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef COMPUTEDIVMESHVELOCITY_H
#define COMPUTEDIVMESHVELOCITY_H

#include <FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
typedef std::vector<Part*> PartVector;
}
}

namespace sierra {
namespace nalu {

class ComputeDivMeshVelocity
{

public:
  ComputeDivMeshVelocity();
  ~ComputeDivMeshVelocity();

  void execute(stk::mesh::PartVector & partVec);
};
    
}
}

#endif /* COMPUTEDIVMESHVELOCITY_H */
