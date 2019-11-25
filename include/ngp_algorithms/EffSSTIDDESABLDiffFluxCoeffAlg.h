/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef EffSSTIDDESABLDiffFluxCoeffAlg_h
#define EffSSTIDDESABLDiffFluxCoeffAlg_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class EffSSTIDDESABLDiffFluxCoeffAlg : public Algorithm
{
public:
  using DblType = double;

  EffSSTIDDESABLDiffFluxCoeffAlg(
    Realm&,
    stk::mesh::Part*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    const double sigmaOne,
    const double sigmaTwo,
    const double sigmaABL);

  virtual ~EffSSTIDDESABLDiffFluxCoeffAlg() = default;
  
  virtual void execute() override;

private:
  ScalarFieldType* viscField_ {nullptr};
  ScalarFieldType* wallDistField_ {nullptr};    
  unsigned visc_ {stk::mesh::InvalidOrdinal};
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned evisc_ {stk::mesh::InvalidOrdinal};
  unsigned fOneBlend_ {stk::mesh::InvalidOrdinal};
  unsigned wallDist_ {stk::mesh::InvalidOrdinal};
  const DblType sigmaOne_;
  const DblType sigmaTwo_;
  const DblType sigmaABL_;
  const DblType abl_bndtw_;
  const DblType abl_deltandtw_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
