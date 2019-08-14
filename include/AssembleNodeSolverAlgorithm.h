/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleNodeSolverAlgorithm_h
#define AssembleNodeSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>

namespace stk {
namespace mesh {
class Part;
}
}

namespace sierra{
namespace nalu{

class Realm;

class AssembleNodeSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleNodeSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem);

  virtual ~AssembleNodeSolverAlgorithm() = default;
  virtual void initialize_connectivity();
  virtual void execute();

  const int sizeOfSystem_;
};

} // namespace nalu
} // namespace Sierra

#endif
