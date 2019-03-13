/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef EffectiveDESABLDiffFluxCoeffAlgorithm_h
#define EffectiveDESABLDiffFluxCoeffAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class EffectiveDESABLDiffFluxCoeffAlgorithm : public Algorithm
{
public:

  EffectiveDESABLDiffFluxCoeffAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *visc,
    ScalarFieldType *tvisc,
    ScalarFieldType *evisc,
    const double sigmaLam,
    const double sigmaTurb,
    const double sigmaOne,
    const double sigmaTwo);
  virtual ~EffectiveDESABLDiffFluxCoeffAlgorithm() {}
  virtual void execute();

  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarFieldType *fOneBlend_;
  ScalarFieldType *fABLBlending_;
  const double sigmaLam_;
  const double sigmaTurb_;
  const double sigmaOne_;
  const double sigmaTwo_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
