/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TurbViscKsgsAlg_h
#define TurbViscKsgsAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra{
namespace nalu{

class Realm;

class TurbViscKsgsAlg : public Algorithm
{
public:
  using DblType = double;

  TurbViscKsgsAlg(
    Realm &realm,
    stk::mesh::Part* part,
    ScalarFieldType* tke,
    ScalarFieldType* density,
    ScalarFieldType* tvisc,
    ScalarFieldType* dualNodalVolume
);

  virtual ~TurbViscKsgsAlg() = default;

  virtual void execute() override;

private:
  ScalarFieldType* tviscField_ {nullptr};
  unsigned tke_  {stk::mesh::InvalidOrdinal};
  unsigned density_  {stk::mesh::InvalidOrdinal};
  unsigned tvisc_  {stk::mesh::InvalidOrdinal};
  unsigned dualNodalVolume_  {stk::mesh::InvalidOrdinal};

  const DblType cmuEps_;
};

} // namespace nalu
} // namespace Sierra

#endif
