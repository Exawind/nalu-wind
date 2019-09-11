/*------------------------------------------------------------------------*/
/*  Copyright 2019 National Renewable Energy Laboratory.                  */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TurbViscSSTAlg_h
#define TurbViscSSTAlg_h

#include "Algorithm.h"
#include "FieldTypeDef.h"

#include "stk_mesh/base/Types.hpp"

namespace sierra{
namespace nalu{

class Realm;

class TurbViscSSTAlg : public Algorithm
{
public:
  using DblType = double;

  TurbViscSSTAlg(
    Realm &realm,
    stk::mesh::Part* part,
    ScalarFieldType* tvisc);

  virtual ~TurbViscSSTAlg() = default;

  virtual void execute() override;

private:
  ScalarFieldType* tviscField_ {nullptr};
  unsigned density_  {stk::mesh::InvalidOrdinal};
  unsigned viscosity_  {stk::mesh::InvalidOrdinal};
  unsigned tke_  {stk::mesh::InvalidOrdinal};
  unsigned sdr_  {stk::mesh::InvalidOrdinal};
  unsigned minDistance_  {stk::mesh::InvalidOrdinal};
  unsigned dudx_  {stk::mesh::InvalidOrdinal};
  unsigned tvisc_  {stk::mesh::InvalidOrdinal};

  const DblType aOne_;
  const DblType betaStar_;
};

} // namespace nalu
} // namespace Sierra

#endif
