// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

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
