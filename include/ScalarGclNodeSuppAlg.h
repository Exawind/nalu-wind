/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ScalarGclNodeSuppAlg_h
#define ScalarGclNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ScalarGclNodeSuppAlg : public SupplementalAlgorithm
{
public:

  ScalarGclNodeSuppAlg(
    ScalarFieldType *scalarQNp1,
    Realm &realm);

  virtual ~ScalarGclNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  ScalarFieldType *scalarQNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *divV_;
  ScalarFieldType *dualNdVolNm1_;
  ScalarFieldType *dualNdVolN_;
  ScalarFieldType *dualNdVolNp1_;
  double dt_;
  double gamma1_;
  double gamma2_;
  double gamma3_;
};

} // namespace nalu
} // namespace Sierra

#endif
