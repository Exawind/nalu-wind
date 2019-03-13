/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergyDESABLNodeSourceSuppAlg_h
#define TurbKineticEnergyDESABLNodeSourceSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class TurbKineticEnergyDESABLNodeSourceSuppAlg : public SupplementalAlgorithm
{
public:
  TurbKineticEnergyDESABLNodeSourceSuppAlg(
    Realm &realm);

  virtual ~TurbKineticEnergyDESABLNodeSourceSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  const double betaStar_;
  ScalarFieldType *tkeNp1_;
  ScalarFieldType *sdrNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *tvisc_;
  GenericFieldType *dudx_;
  ScalarFieldType *dualNodalVolume_;
  ScalarFieldType *maxLengthScale_;
  ScalarFieldType *fOneBlend_;
  ScalarFieldType *fABLBlending_;
  double tkeProdLimitRatio_;
  int nDim_;
  double cDESke_;
  double cDESkw_;
  double cEps_;

};

} // namespace nalu
} // namespace Sierra

#endif
