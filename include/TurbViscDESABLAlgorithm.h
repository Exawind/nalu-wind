/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbViscDESABLAlgorithm_h
#define TurbViscDESABLAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class TurbViscDESABLAlgorithm : public Algorithm
{
public:
  
  TurbViscDESABLAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~TurbViscDESABLAlgorithm() {}
  virtual void execute();

  const double aOne_;
  const double betaStar_;
  const double cmuEps_;

  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  ScalarFieldType *tke_;
  ScalarFieldType *sdr_;
  ScalarFieldType *minDistance_;
  GenericFieldType *dudx_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *fABLBlending_;
  ScalarFieldType *dualNodalVolume_;
};

} // namespace nalu
} // namespace Sierra

#endif
