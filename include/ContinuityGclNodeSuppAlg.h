/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef ContinuityGclNodeSuppAlg_h
#define ContinuityGclNodeSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class ContinuityGclNodeSuppAlg : public SupplementalAlgorithm
{
public:

  ContinuityGclNodeSuppAlg(
    Realm &realm);

  virtual ~ContinuityGclNodeSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
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
