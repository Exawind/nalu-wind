/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef EffSSTDiffFluxCoeffAlg_h
#define EffSSTDiffFluxCoeffAlg_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class EffSSTDiffFluxCoeffAlg : public Algorithm
{
public:
  using DblType = double;

  EffSSTDiffFluxCoeffAlg(
    Realm&,
    stk::mesh::Part*,
    ScalarFieldType*,
    ScalarFieldType*,
    ScalarFieldType*,
    const double sigmaOne,
    const double sigmaTwo);

  virtual ~EffSSTDiffFluxCoeffAlg() = default;
  
  virtual void execute() override;

private:
  ScalarFieldType* viscField_ {nullptr};
  unsigned visc_ {stk::mesh::InvalidOrdinal};
  unsigned tvisc_ {stk::mesh::InvalidOrdinal};
  unsigned evisc_ {stk::mesh::InvalidOrdinal};
  unsigned fOneBlend_ {stk::mesh::InvalidOrdinal};
  const DblType sigmaOne_;
  const DblType sigmaTwo_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
