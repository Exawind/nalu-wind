/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbViscSSTAlgorithm_h
#define TurbViscSSTAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class TurbViscSSTAlgorithm : public Algorithm
{
public:
  
  TurbViscSSTAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~TurbViscSSTAlgorithm() {}
  virtual void execute();

  const double aOne_;
  const double betaStar_;

  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  ScalarFieldType *tke_;
  ScalarFieldType *sdr_;
  ScalarFieldType *minDistance_;
  GenericFieldType *dudx_;
  ScalarFieldType *tvisc_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
