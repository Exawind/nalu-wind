/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleMomentumEdgeABLTopBC_h
#define AssembleMomentumEdgeABLTopBC_h

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

class AssembleMomentumEdgeABLTopBC : public SolverAlgorithm
{
public:

  AssembleMomentumEdgeABLTopBC(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem, std::vector<int>& grid_dims);
  virtual ~AssembleMomentumEdgeABLTopBC() {}
  virtual void initialize_connectivity();
  virtual void execute();

  VectorFieldType *velocity_;
  VectorFieldType *bcVelocity_;
  ScalarFieldType *density_;
  GenericFieldType *exposedAreaVec_;
  int imax_, jmax_, kmax_;
  std::vector<double> sampleVel_;
};

} // namespace nalu
} // namespace Sierra

#endif
